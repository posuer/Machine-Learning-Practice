# 已核实的公开资产与改动来源

## 训练代码（真实上游，不是虚构的“已有代码库”）

- [Hugging Face TRL 仓库](https://github.com/huggingface/trl)
- [官方 VLM SFT 示例](https://github.com/huggingface/trl/blob/accf7383a33c618a2edc7205107120b5f32e28a3/examples/scripts/sft_vlm.py)
- 版本：v0.19.1，commit `accf7383a33c618a2edc7205107120b5f32e28a3`。
- `upstream/sft_vlm.py` 是下载的未修改原版，保留 Apache-2.0 copyright；完整许可证为 `upstream/LICENSE-TRL`，来源与 SHA-256 记录在 `upstream/provenance.json`。
- 上游示例的默认运行说明测试于 8×H100，不能原样当作 T4 配置。本项目借用 SFTTrainer/VLM collator 流程，增加 Qwen 图片处理、answer-only masking、4-bit LoRA、保守资源配置和保存重载。
- `receipt_lab/train.py` 调用真实 TRL/PEFT，不自写替代训练器。数据映射、评估、API 和演练材料是本次为练习新增的代码，不是 Reducto 官方代码。

## 模型

- [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- 固定 revision：`66285546d2b821cf421d4f5eb2576359d3770cd3`。
- [模型使用说明](https://github.com/QwenLM/Qwen2.5-VL)。用 Transformers 原生 Qwen2_5_VLForConditionalGeneration，无需执行远端自定义代码。
- 模型仓库标注许可证为 [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE)，不是 Apache-2.0；本练习是面试学习用途。真实商业部署需自行核对模型条款。
- 这里只固定和提供真实模型的下载/加载入口，不把大权重提交 Git。

## 数据

- [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- 固定 revision：`7f0115a4b758a71d6473b8d085751692da2fef98`。
- [CORD 原始项目与字段定义](https://github.com/clovaai/cord)，许可 CC-BY-4.0；请保留来源。
- CORD 论文：Consolidated Receipt Dataset for Post-OCR Parsing，Park et al.。公开发布为 1,000 张印度尼西亚收据，不要将论文完整采集规模误写为公开样本数。
- 本项目投影目标：menu.nm/cnt/price → items；sub_total.subtotal_price/tax_price → subtotal/tax；total.total_price → total。忽略 submenu、void、现金与找零等字段，显式记录为自定义任务。多值字段/多组小计无法投影到本题单值 schema 时，记录排除清单而不任意选择或拼接。
- JSON 里的 OCR 文本和框不传给推理模型；它们属于可选数据分析信息。训练答案来自人工标注的 gt_parse。

## 与 Reducto 数据包的关系

[reducto/mle-interview](https://huggingface.co/datasets/reducto/mle-interview) 的公开包只有 996 张完整页面，没有本任务所需的监督答案。因此主训练/评估使用 CORD；Reducto 包仅可用于域外探索或另行标注的扩展。

不宣称这是 Reducto onsite 的真实模型、数据、题目或评估标准。
