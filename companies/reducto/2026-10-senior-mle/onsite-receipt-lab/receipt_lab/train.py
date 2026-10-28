# Adapted from Hugging Face TRL examples/scripts/sft_vlm.py (Apache-2.0).
# See upstream/ for the unmodified source, license and pinned provenance.
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from .common import load_config, load_rows
from .modeling import ReceiptCollator, compute_dtype, load_model, processor_for, tiny_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/cord/train.jsonl")
    p.add_argument("--model-config", default="configs/baseline.json")
    p.add_argument("--train-config", default="configs/train_t4.json")
    p.add_argument("--output", default="runs/lora")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--smoke-model", action="store_true", help="Tiny random Qwen; no real baseline quality")
    args = p.parse_args()
    cfg, train_cfg = load_config(args.model_config), load_config(args.train_config)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory is not empty; preserve previous experiments and choose another.")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    if args.smoke_model:
        cfg.update(min_pixels=3136, max_pixels=3136, load_in_4bit=False, max_new_tokens=16)
    elif not torch.cuda.is_available():
        raise SystemExit("Use a CUDA GPU for the real QLoRA run; CPU supports only --smoke-model validation.")
    processor = processor_for(cfg)
    if args.smoke_model:
        torch.set_num_threads(2)
        model = tiny_model(processor)
        base = out / "tiny-base"
        model.save_pretrained(base)
        cfg.update(model_id=str(base), model_revision=None, smoke_model=True)
    else:
        model = load_model(cfg)
        if cfg["load_in_4bit"]:
            model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    rows = load_rows(args.manifest, args.limit)
    # Freeze base weights; adapt language attention projections. Vision qkv uses different names.
    lora = LoraConfig(r=train_cfg["lora_r"], lora_alpha=train_cfg["lora_alpha"],
                      lora_dropout=train_cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    dtype = compute_dtype()
    training = SFTConfig(
        output_dir=str(out), max_steps=args.max_steps or train_cfg["max_steps"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=1 if args.smoke_model else train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"], logging_steps=1 if args.smoke_model else train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"], save_total_limit=2,
        fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", max_grad_norm=1.0, warmup_ratio=0.05,
        remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None, packing=False, report_to="none", seed=cfg["seed"],
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, args=training, train_dataset=Dataset.from_list(rows),
                         processing_class=processor.tokenizer,
                         data_collator=ReceiptCollator(processor, cfg["max_sequence_tokens"]),
                         peft_config=lora)
    result = trainer.train()
    trainer.save_model(str(out))
    processor.save_pretrained(out)
    trainer.save_state()
    (out / "model_config.json").write_text(json.dumps(cfg, indent=2))
    (out / "run.json").write_text(json.dumps({"train_config": train_cfg, "arguments": vars(args),
        "metrics": result.metrics, "train_ids": [r["id"] for r in rows],
        "smoke_only": args.smoke_model}, indent=2))
    print(f"Saved adapter and processor to {out}")


if __name__ == "__main__":
    main()
