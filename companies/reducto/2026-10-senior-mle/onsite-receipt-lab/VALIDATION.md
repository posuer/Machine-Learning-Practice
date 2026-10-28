# 交付验证记录

验证日期：2026-10-28。当前机器只有 CPU，Python 3.12、torch 2.6.0+cpu、torchvision 0.21.0+cpu，其余使用 requirements.txt 中的固定版本。

## 已实际完成

- 下载并处理固定 revision 的全部 CORD-v2：原始 800/100/100；适合本题单值 schema 的 train/validation/test 分别为 **782/99/98**。21 条多值标注写入 excluded.jsonl，没有猜测目标。总体覆盖率 97.9%。
- 在可用子集中发现两组完全相同的训练图片，均位于 train 内；没有发现跨 split 的精确图片重复。这里只检查解码后像素相同，不能排除近似重复或模板泄漏。
- 官方 TRL 原版训练示例和 LICENSE 已下载，SHA-256 与 upstream/provenance.json 一致。
- 10 项自动化检查通过：字段映射、完美预测、空白处理、错误金额、非法/截断输出、多值标注、重复 JSON key、行顺序、缺失预测分母、接口输入校验。
- 真实 Qwen processor 与真实收据通过 collator 检查：prompt、图片 token、padding 不计 loss；解码监督 token 得到完整目标 JSON，包含 assistant 结束 token。
- 缩小的随机 Qwen2.5-VL 架构通过真实 TRL/PEFT 执行 2 个 optimizer steps。loss 有限；保存的 8 个 LoRA B 张量均有限，且存在非零更新。
- 在另一个 CLI 进程重载基础模型和 adapter，完成 2 条真实 validation 图片的生成、评估和静态 HTML 报告。
- 使用重载后的 tiny-model 和 FastAPI TestClient 验证接口：图片请求返回 200 / needs_review；非法图片返回 400；health 返回 ready。此项不是负载测试。
- Colab notebook 的 Python 单元格通过语法检查；未在 Colab runtime 执行整本 notebook。

## 可复现命令与产物

工作目录为本项目目录，使用 `/workspace/reducto-onsite-runtime/venv/bin/python`，HF_HOME 为 `/workspace/reducto-onsite-runtime/hf-cache`。

```bash
python -m pytest -q tests
python -m receipt_lab.prepare --output data/cord
python -m receipt_lab.prepare --limit-per-split 2 --output data/smoke
python -m receipt_lab.train --smoke-model --manifest data/smoke/train.jsonl --limit 2 --max-steps 2 --output runs/tiny-smoke-v2
python -m receipt_lab.predict --manifest data/smoke/validation.jsonl --adapter runs/tiny-smoke-v2 --output runs/tiny-smoke-v2.predictions.jsonl
python -m receipt_lab.evaluate --manifest data/smoke/validation.jsonl --predictions runs/tiny-smoke-v2.predictions.jsonl --output runs/tiny-smoke-v2.metrics.json
python -m receipt_lab.report --manifest data/smoke/validation.jsonl --baseline runs/tiny-smoke-v2.predictions.jsonl --output runs/tiny-smoke-v2.review.html
```

这些输出在当前 workspace 已存在；重跑时使用新的目录/文件名，或者复用已经准备好的数据。数据、模型缓存、tiny 权重和 runs 不打包进 Colab starter，也不提交 Git。

tiny 模型为随机初始化，两个输出均为截断、无效 JSON，指标为 0。这只说明失败能够被记录和计分，**不是预训练 3B 模型的 baseline 分数**。

## 仍须在 GPU 上验证

真实 3B 权重尚未在本机下载或运行；已经下载的是 processor/config 和真实数据。Colab 的 prefetch 单元格会下载固定 revision 的完整预训练权重。

4-bit bitsandbytes 加载、T4 FP16 / 支持设备的 BF16、真实 3B LoRA 前后向、显存峰值、训练耗时、生成质量和吞吐量都尚未验证。先执行 notebook 中的 baseline smoke 和 2-step GPU smoke，再做完整实验。不能依据 CPU tiny 测试声称 16GB 一定够用或某个改进有效。
