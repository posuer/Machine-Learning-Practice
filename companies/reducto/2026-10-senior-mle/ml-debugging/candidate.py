"""VLM 训练调试题：请结合同目录 README.md 审查并修正 train_step。

本文件保留原始待调试代码，不是正确实现；注释解释实际执行的操作，
不代表这些操作已经满足题目的训练要求。此文件不包含模型、数据或运行入口。

背景：原始样本可以分别存放 prompt 和 answer。经过预处理后，本题的
input_ids 已经包含拼接后的 prompt、正确答案及特殊 token。
训练时提供正确答案作为后续位置的上下文，称为 teacher forcing。
模型内部的 causal mask 阻止当前位置读取未来文本。

符号：B = batch 大小；T = 补齐后的文本长度；V = 词表大小。
注意三种不同职责：
  attention_mask：标记真实输入与 padding；本题中不是三角形 causal mask。
  模型内部的 causal mask：限制文本 attention 只能读取当前和之前的位置。
  labels：指定监督目标，值为 -100 的目标可由 loss 忽略。
"""

# PyTorch：提供 Tensor、自动求导以及 softmax 等张量运算。
import torch
# functional 提供交叉熵等函数式 API；F 是常用别名。
import torch.nn.functional as F


# model：已经创建的视觉语言模型，输出对象包含 .logits。
# batch：已经预处理的一个批次，含图片、token、attention_mask 和 labels。
# optimizer：已绑定模型参数的优化器，例如 SGD 或 AdamW。
# 题目要求一次调用完成一次独立更新，并返回 Python float 类型的 loss。
def train_step(model, batch, optimizer):
    # 切换为训练模式，影响 Dropout、BatchNorm 等具有训练/评估差异的模块。
    # 这一步本身不会执行训练、更新参数，也不会清空已有梯度。
    model.train()

    # 页面图像，shape 为 (B, C, H, W)，本题 dtype 为 float32。
    # .cuda() 将 Tensor 放到 CUDA 设备上（默认使用当前 CUDA 设备）。
    # 返回的 Tensor 绑定到局部变量，不会直接替换 batch 字典中的值。
    images = batch["pixel_values"].cuda()
    # 完整文本的 token ID，shape 为 (B, T)，dtype 为 torch.long。
    # 元素是词表索引，不是 embedding，也不是概率；有效 ID 范围为 [0, V)。
    input_ids = batch["input_ids"].cuda()
    # shape 为 (B, T)：1 表示有效文本，0 表示补齐长度的 padding。
    # prompt 和 answer 都属于有效文本；具体 attention 处理由模型内部完成。
    attention_mask = batch["attention_mask"].cuda()

    # clone() 复制 input_ids 的数值，创建独立存储的 Tensor，保持 shape/dtype/device。
    # 因而此处 labels 的内容与 input_ids 一样；这行没有读取 batch["labels"]。
    # 题目提供的 batch["labels"] 是预处理后的监督目标，说明见 README。
    labels = input_ids.clone()

    # 一次前向传播：模型结合图像与文本，为每个文本位置计算预测分数。
    # 本题使用普通梯度模式，PyTorch 会记录涉及可训练参数的运算供反向传播使用。
    # 这里没有向模型传入 labels，因此按题目约定，模型只返回预测、不计算 loss。
    outputs = model(
        pixel_values=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # logits 是未归一化分数，可以为正或负，各类别之和不要求等于 1。
    # logits[b, t, :] 含 V 个分数，表示模型在位置 t 对下一个 token 的预测。
    logits = outputs.logits  # (B, T, V)

    # 三个索引依次对应 batch、文本位置和词表维度。
    # :-1 保留从开头到倒数第二个位置，去掉最后一个位置。
    # 切片后 shape 为 (B, T-1, V)，没有重新计算或移动任何预测分数。
    shift_logits = logits[:, :-1, :]
    # 同样截去 labels 最后一个位置，得到 (B, T-1)。
    # 变量名中的 shift 不会自动执行位移；实际保留哪些目标完全由切片决定。
    shift_labels = labels[:, :-1]

    # 在最后一维（词表维度 V）做 softmax，将每个位置的 V 个分数转成概率。
    # 每组概率之和约为 1，输出 shape 仍为 (B, T-1, V)。
    # softmax 可微，返回的 probs 仍然连接在计算图中。
    probs = torch.softmax(shift_logits, dim=-1)

    # cross_entropy 将每个位置视为一个 V 分类任务。
    # 对整数类别标签，该 API 的第一个参数约定是未归一化 logits，
    # 内部通过 log_softmax 和负对数似然计算损失；第二个参数是目标类别索引。
    # 此处实际传入的是上面得到的 probs 和 shift_labels，保留题目原始写法。
    # 默认 ignore_index=-100：忽略目标标签为 -100 的项。
    # 默认 reduction="mean"：无类别权重时，对未忽略的目标项取平均，返回标量。
    loss = F.cross_entropy(
        # 将 batch 和时间维合并：(B, T-1, V) -> (B*(T-1), V)。
        # 每一行对应一个位置的 V 个类别值，probs.shape[-1] 就是 V。
        # reshape 按需返回视图或创建副本，不改变自动求导关系。
        probs.reshape(-1, probs.shape[-1]),
        # (B, T-1) -> (B*(T-1),)，与上方展平后的每一行逐项配对。
        # -1 表示让 PyTorch 根据元素总数推断这一维的长度。
        shift_labels.reshape(-1),
    )

    # 从标量 loss 沿计算图反向求导，将参数梯度累加到 parameter.grad。
    # backward() 本身不更新参数；已有 .grad 不会被它自动清零。
    loss.backward()
    # 优化器根据参数当前的 .grad 和自身状态执行更新。
    # 例如 SGD 使用学习率，AdamW 还维护动量等状态；step() 不会自动清空梯度。
    optimizer.step()

    # 将单元素 Tensor 转成 Python 数值，便于日志记录；这个数值不再参与求导。
    # 返回的是前向传播时算出的 loss，并不是更新参数后重新计算的 loss。
    # 如果 loss 在 GPU 上，读取 .item() 通常需要等待对应 GPU 运算完成。
    return loss.item()
