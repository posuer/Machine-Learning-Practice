# 手写练习：实现一个 VLM 的单步 SFT 训练函数。
# 本文件只有接口、任务说明和待完成步骤，不包含参考实现。
# 完整背景见同目录 README.md；练习时可以先不打开 reference.py。
#
# 输入约定：
#   model：单设备上的模型，内部已实现 causal attention。
#   optimizer：已绑定所有需要更新的模型参数。
#   batch：包含以下 Tensor，最初可以位于 CPU：
#       pixel_values   float32，(B, C, H, W)，预处理后的页面图片。
#       input_ids      long，(B, T)，拼接后的 prompt、答案和特殊 token。
#       attention_mask long，(B, T)，1 为有效文本，0 为 padding。
#       labels         long，(B, T)，与原始 input_ids 位置对齐。
#                      prompt/padding 标签为 -100，其余为目标 token ID。
#
# 模型接口：
#   接收 pixel_values、input_ids、attention_mask 三个命名参数。
#   返回对象的 .logits 为 (B, T, V)，内容是未归一化分数。
#   位置 t 的输出用于预测下一个 token；模型不自动计算本题的 loss。
#
# 目标：
#   - 每次调用独立更新一次参数，不使用梯度累积。
#   - 支持 CPU 或单张 GPU，使用普通 FP32，不修改原始 batch。
#   - 对参与监督的 next-token 目标计算平均交叉熵。
#   - 返回本次参数更新之前的 loss，类型为 Python float。
#   - 进阶要求：若对齐后没有有效监督目标，抛出 ValueError，不更新参数。

# TODO：添加你需要的 import。
import torch.nn.functional as F

def train_step(model, batch, optimizer):
    # 1. 设置模型的训练状态，确定本次计算所使用的设备。
    # TODO
    model.train()

    # 2. 从 batch 准备图片、文本、attention mask 和监督标签。
    #    检查设备、dtype 和监督信息是否符合题意。
    # TODO
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    # 3. 为本次独立的参数更新准备梯度状态。
    # TODO
    optimizer.zero_grad(set_to_none=True)

    # 4. 调用模型完成前向传播，取出各位置的预测分数。
    # TODO
    output = model(
      pixel_values=pixel_values,
      input_ids=input_ids,
      attention_mask=attention_mask
      )
    logits = output.logits
    # 5. 将预测和 next-token 监督目标对齐。
    #    自己写出切片，并确认处理后两者的 shape 和位置关系。
    # TODO
    logits_shifted = logits[:,:-1,:]
    labels_shifted = labels[:,1:]
    
    # 6. 进阶：检查有效监督目标是否存在；若不存在，明确报错。
    #    此检查也可以移到前向传播之前，避免无效计算。
    # TODO

    # 7. 整理预测与目标的维度，计算平均交叉熵。
    #    遵循 loss API 的输入约定和题目的监督范围。
    # TODO
    loss = F.cross_entropy(logits_shifted.reshape(-1, logits_shifted.shape[-1]), labels_shifted.reshape(-1), ignore_index=-100, reduction="mean")
    # 8. 计算梯度，并执行一次参数更新。
    # TODO
    loss.backward()
    optimizer.step()
    # 9. 返回适合记录日志的 Python float。
    # TODO
    return loss.item()
    # 完成上面的实现后删除此行；未完成时避免被误认为可用的训练函数。
    # raise NotImplementedError("请完成 train_step 的手写实现。")
