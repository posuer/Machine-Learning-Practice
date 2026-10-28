import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import set_seed

from .common import load_config, load_rows, messages, strict_json, validate_schema
from .modeling import load_model, processor_for


class Extractor:
    def __init__(self, config_path="configs/baseline.json", adapter=None):
        self.config = load_config(Path(adapter) / "model_config.json" if adapter else config_path)
        set_seed(self.config["seed"])
        self.processor = processor_for(self.config, source=adapter)
        self.model = load_model(self.config)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.model.config.use_cache = True

    def predict(self, image):
        # This measures preprocess+generate+decode, excluding model loading and file/network IO.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        prompt = self.processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt")
        batch = batch.to(next(self.model.parameters()).device)
        with torch.inference_mode():
            output = self.model.generate(**batch, max_new_tokens=self.config["max_new_tokens"],
                do_sample=False, use_cache=True)
        generated = output[0, batch["input_ids"].shape[1]:]
        text = self.processor.decode(generated, skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended = len(generated) > 0 and int(generated[-1]) in eos_ids
        truncated = len(generated) >= self.config["max_new_tokens"] and not ended
        try:
            parsed = strict_json(text)
            valid_json, valid_schema = True, validate_schema(parsed)
        except (ValueError, TypeError):
            parsed, valid_json, valid_schema = None, False, False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"raw_text": text, "parsed": parsed, "json_valid": valid_json,
                "schema_valid": valid_schema, "truncated": truncated,
                "status": "ok" if valid_schema and not truncated else "needs_review",
                "generated_tokens": len(generated), "latency_seconds": time.perf_counter() - start}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/cord/validation.jsonl")
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--adapter")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = load_rows(args.manifest, args.limit)
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Prediction file already exists; choose a new run name.")
    output.parent.mkdir(parents=True, exist_ok=True)
    extractor = Extractor(args.config, args.adapter)
    with output.open("x") as f:
        for index, row in enumerate(rows):
            start = time.perf_counter()
            try:
                with Image.open(row["image_path"]) as im:
                    prediction = extractor.predict(im)
            except torch.cuda.OutOfMemoryError:
                # Do not repeatedly run a bad resource configuration or conceal incomplete coverage.
                raise
            except (OSError, ValueError) as exc:
                prediction = {"raw_text": "", "status": "error", "error": str(exc),
                              "latency_seconds": time.perf_counter() - start, "truncated": False}
            f.write(json.dumps({"id": row["id"], **prediction}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{index+1}/{len(rows)} {row['id']}: {prediction['status']}", flush=True)
    output.with_suffix(".run.json").write_text(json.dumps({"config": extractor.config,
        "arguments": vars(args), "count": len(rows), "first_request_includes_warmup": True}, indent=2))


if __name__ == "__main__":
    main()
