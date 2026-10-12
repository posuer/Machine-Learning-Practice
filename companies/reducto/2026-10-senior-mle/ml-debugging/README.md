# ML Debugging：文档解析 VLM 的训练步骤

**形式：** 模拟现场面试，阅读代码、解释判断、修改实现并设计验证。
**建议时长：** 25 分钟完成核心任务；另用 10–15 分钟讨论验证和追问。
**实现语言：** Python / PyTorch。

本题根据你提供的准备材料整理，是针对岗位的模拟练习，不是已确认的 Reducto 真题。此文件只包含候选人可见的题目说明，不包含参考答案或缺陷清单。

## 1. 业务背景

团队正在训练一个视觉语言模型（Vision-Language Model，VLM）：模型读取一张文档页面图片和一段文字指令，输出页面对应的结构化文本。

例如，一张发票图片包含商家、日期和金额，用户给出的指令是：

```text
读取这张发票，返回包含 vendor、date 和 total 的 JSON。
```

期望模型输出的文本可能是：

```json
{"vendor": "Example Store", "date": "2026-09-30", "total": "42.00"}
```

这里的 JSON 只是解释业务的例子。本题不要求实现 OCR、图像编码器、JSON 解析器，也不要求下载或训练真实的大模型。你的工作范围是检查一个训练步骤：它是否正确地利用已经准备好的数据更新模型参数。

## 2. 模型如何学习这个任务

训练数据里，每张图片都配有文字指令和人工确认的正确答案。文字会经过 tokenizer，转换成整数 token ID；一个 token 可能是字、词片段、标点或特殊符号，不一定是一个完整的单词。

本题采用自回归生成：模型根据图片和之前的文字，预测下一个 token。训练时会把正确答案作为输入序列的一部分，这种方式称为 teacher forcing。模型内部的 causal attention 保证当前位置只能使用当前及之前的文本，不能看到未来答案。

推理时没有正确答案可供输入，模型需要逐步生成；本题只实现训练步骤，不实现推理解码。

## 3. 已提供的数据

`batch` 是一个字典。预处理、tokenization 和批次组装已经完成，你不需要重新实现它们。

符号定义：

| 符号 | 含义 |
| --- | --- |
| `B` | 一个 batch 中的样本数 |
| `T` | padding 后的文本序列长度 |
| `V` | 词表大小，合法 token ID 范围为 `[0, V)` |
| `C, H, W` | 图像的通道数、高度和宽度 |

输入字段：

| 字段 | Shape | dtype | 含义 |
| --- | --- | --- | --- |
| `pixel_values` | `(B, C, H, W)` | `torch.float32` | 已经预处理的页面图片 |
| `input_ids` | `(B, T)` | `torch.long` | 包含特殊 token、prompt 和正确答案的完整文本序列 |
| `attention_mask` | `(B, T)` | `torch.long` | `1` 表示真实文本位置，`0` 表示 padding |
| `labels` | `(B, T)` | `torch.long` | 与原始输入位置对齐的监督标签；不参与训练评分的位置为 `-100` |

不同样本长度可能不同，本题采用右侧 padding。`attention_mask` 用来描述输入中哪些位置有效；`labels` 用来定义哪些目标参与 loss。两者的职责不同。

### 一个序列的示意

以下符号代表 token，而不是实际的字符串输入：

```text
位置                 0       1       2       3       4       5       6
input_ids          [BOS,    P_1,    P_2,    A_1,    A_2,    EOS,    PAD]
attention_mask     [  1,      1,      1,      1,      1,      1,      0]
labels             [-100,  -100,   -100,    A_1,    A_2,    EOS,   -100]
```

- `BOS` 表示序列开始，`EOS` 表示答案结束，`PAD` 用于补齐长度。
- `P_1, P_2` 是 prompt 中的 token；`A_1, A_2` 是答案中的 token。
- `labels` 中非负的值是目标 token ID；`-100` 不是词表中的 token，而是忽略监督的标记。
- 数据处理已经将 prompt 和 padding 对应的标签设为 `-100`，答案和结束标记参与监督。
- 这里的 `labels` 尚未做 next-token 对齐；它与 `input_ids` 使用同一套原始位置索引。

## 4. 模型与优化器接口

调用方式：

```python
outputs = model(
    pixel_values=pixel_values,
    input_ids=input_ids,
    attention_mask=attention_mask,
)
logits = outputs.logits
```

约定如下：

- `logits` 的 shape 是 `(B, T, V)`，每个元素是未归一化的分数。
- `logits[b, t, :]` 表示样本 `b` 在看到图片及文本位置 `0..t` 后，对下一个 token 的预测分数。
- 本题的模型输出与文本长度 `T` 对齐。图片特征已在模型内部处理，不会在输出中增加额外的文本位置。
- 模型内部已经实现文本的因果约束和 padding attention 处理。
- 本次调用不传入 `labels`，模型不替你计算 loss，也不替你对齐预测与目标。
- 模型参数位于同一个设备上。输入 batch 最初可以位于 CPU，需要支持模型在 CPU 或单张 GPU 上运行。
- `optimizer` 已经创建，并绑定了需要更新的模型参数。

## 5. 问题陈述

同事提供了 [candidate.py](candidate.py) 中的 `train_step`。这段代码尚未通过正确性审查。团队发现训练效果不符合预期，但目前没有足够的日志证明具体原因。

请审查并修正这个函数，使其满足以下训练目标和执行约定：

1. 根据图片和 prompt，学习生成正确答案，以自回归 token 分类的交叉熵作为训练目标。
2. 尊重数据处理中定义的监督范围。
3. 每次函数调用执行一次独立的参数更新；本题不采用梯度累积。
4. 支持 CPU 和单 GPU，不修改调用者传入的 batch 内容。
5. 返回一个 Python `float`，表示本次参数更新前的训练 loss。

本题使用普通 FP32 训练，不要求 AMP、分布式训练、学习率调度或梯度裁剪。核心任务可以假设每个 batch 在有效 next-token 对齐后至少有一个参与监督的 token；如果时间允许，再讨论不满足此条件时应采用什么策略。

## 6. 你需要交付什么

- **审查说明：** 逐项解释你认为存在的问题，说明理由及其可能影响；区分确定的问题与需要更多证据的猜测。
- **代码修改：** 在 `candidate.py` 中完成可供模型调用的训练函数。
- **验证方案：** 说明如何验证训练目标、参数更新及设备兼容性，而不只是确认代码没有抛异常。
- **后续排查：** 假设修复后训练 loss 下降，但验证集效果仍然较差，解释你下一步会收集什么证据，以及如何安排排查顺序。

你可以查阅 PyTorch API 文档。请边分析边解释；如需补充假设，请明确说出，不必猜测未给出的模型内部实现。

## 7. 练习方式

先阅读本题并用自己的话复述输入、输出和训练目标，再开始审查代码。如果背景或接口不清楚，可以先向面试官提问。

直接运行 `candidate.py` 只会定义函数，不会启动训练，也不能证明实现正确。本目录不提供真实 VLM 或数据加载器；下面的参考验证使用可训练分数表和合成 batch，只验证训练函数的行为。

已有环境的激活方式：

```bash
cd /workspace/Machine-Learning-Practice
source /workspace/ml-interview-env/activate.sh
```

本题的正确性验证不需要下载预训练模型，不需要外部 API，也不要求 GPU。

## 8. 参考实现与验证（自行选择何时查看）

- [reference.py](reference.py)：带详细中文注释的正确实现，原始 candidate.py 保持不变。
- [test_reference.py](test_reference.py)：验证 next-token 对齐、有效 token loss、屏蔽位置的直接 logits 梯度、参数更新、旧梯度清理、batch 不被修改、单 batch 收敛及空监督处理。GPU 检查在无 CUDA 时跳过。

激活环境后运行：

```bash
pytest -q companies/reducto/2026-10-senior-mle/ml-debugging/test_reference.py
```

测试中的分数表不是 VLM，也没有模拟 causal attention 或文档理解。这些测试验证训练步骤，不能证明真实模型无未来信息泄漏或具有泛化能力。
