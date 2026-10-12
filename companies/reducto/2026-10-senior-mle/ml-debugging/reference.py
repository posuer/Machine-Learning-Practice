"""参考实现：FP32、单设备、每次调用独立更新的 VLM/SFT 训练步骤。

对应 README.md 的模型契约：logits 为 (B, T, V)，位置 t 预测下一个 token。
这里只实现训练步骤；真实模型需要自己提供 causal attention，避免读取未来文本。
"""

import torch.nn.functional as F


def train_step(model, batch, optimizer):
    """更新一次参数，返回更新前的平均有效 token loss（Python float）。

    batch 已经过预处理，labels 中 prompt 和 padding 的位置为 -100。
    模型参数位于同一设备，optimizer 绑定了所有需要更新的参数。
    不使用梯度累积、AMP、模型并行；调用时应处于正常的梯度模式。
    """
    # 开启训练行为，例如启用 Dropout。train() 不负责开启 autograd 或清空梯度。
    model.train()

    # 以模型实际所在的设备为准，不硬编码 CUDA，也不根据 GPU 是否存在自行选卡。
    # 对本题的单设备模型，取第一个参数的 device 即可。
    device = next(model.parameters()).device

    # .to(device) 保持 dtype，只把数据放到与模型相同的设备上。
    # 不替换 batch 字典中的值，也不对 Tensor 做原地修改。
    # 已在目标设备上的 Tensor 可能直接返回自身；这里不修改它，因此不需要 clone。
    images = batch["pixel_values"].to(device)        # float32: (B, C, H, W)
    input_ids = batch["input_ids"].to(device)        # long: (B, T)
    attention_mask = batch["attention_mask"].to(device)  # long: (B, T)
    labels = batch["labels"].to(device)             # long: (B, T)

    # 必须保留预处理生成的 labels，而不是从 input_ids 重新复制。
    # input_ids 包含模型需要读取的全部文本；labels 中的 -100 则定义不计分的目标。
    # prompt 可以参与上下文计算，同时不作为需要预测的监督目标。

    # 每个预测位置对应下一个位置的标签，因此目标从原始位置 1 开始。
    # 例如原序列 [BOS, P, A, EOS]：目标标签取位置 [1, 2, 3]。
    # 如 prompt 被屏蔽，则目标可以是 [-100, A, EOS]。
    shift_labels = labels[:, 1:]                    # (B, T-1)

    # mean loss 需要至少一个有效目标。全部是 -100 时不存在可取平均的监督项。
    # 本实现选择明确报错，让调用者修复/过滤数据，不返回假装成功的 0 loss。
    # 长度不足两个 token 的空目标序列，也会触发这个检查。
    if not shift_labels.ne(-100).any().item():
        raise ValueError("Batch has no supervised next-token targets after shifting.")

    # PyTorch 默认将 backward 得到的梯度累加到参数 .grad。
    # 本题每次调用独立更新，所以必须在本次 backward 之前清掉上次的梯度。
    # set_to_none=True 将已有梯度设为 None，通常可减少清零的内存操作。
    optimizer.zero_grad(set_to_none=True)

    # 单次前向传播；没有传 labels，模型不计算 loss，也不替我们 shift 标签。
    # 注意：若改为使用 Hugging Face 模型的 outputs.loss，应遵循其接口，
    # 通常传入未 shift 的 labels，由模型内部完成 shift，不能再重复移位。
    outputs = model(
        pixel_values=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits                         # (B, T, V)，原始分数

    # logits[:, t, :] 预测 labels[:, t+1]。
    # 最后一个 logits 在给定序列中没有下一个标签，所以截掉最后一个预测位置。
    # 与上面的 labels[:, 1:] 配对，两个 Tensor 的时间维长度均为 T-1。
    shift_logits = logits[:, :-1, :]                # (B, T-1, V)

    # cross_entropy 直接接收 logits：内部完成稳定的 log_softmax 与负对数似然。
    # 不先手动 softmax，否则交叉熵会把概率再次当作分数处理，改变目标与梯度。
    # 展平前两维后，每一行是一个 token 预测，目标是对应的整数 token ID。
    # 切片后的 Tensor 可能不连续，reshape 会按需复制，且保留求导关系。
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),  # (B*(T-1), V)
        shift_labels.reshape(-1),                         # (B*(T-1),)
        ignore_index=-100,  # 不计算被屏蔽目标的 loss
        reduction="mean",  # 对全 batch 的有效 token 求平均，而非对每条样本等权平均
    )

    # 沿计算图计算梯度，包括对图像/文本分支中参与计算的可训练参数求导。
    # 输入 token ID 是整数，不需要也不能通过它们本身求梯度；embedding 参数可以。
    loss.backward()

    # 根据本次梯度更新参数及优化器状态。这里不做梯度累积，每次调用更新一次。
    optimizer.step()

    # loss 是更新之前那次 forward 算出的值，step() 不会自动重新计算它。
    # .item() 返回脱离计算图的 Python float，用于日志；在 GPU 上可能导致同步。
    return loss.item()
