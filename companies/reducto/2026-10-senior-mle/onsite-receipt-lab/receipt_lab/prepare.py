"""Download the pinned real CORD-v2 dataset and materialize deterministic manifests."""
import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from .common import validate_schema

DATASET = "naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"


class UnsupportedAnnotation(ValueError):
    """Source annotation cannot be represented by this task's single-value schema."""


def target_from_ground_truth(text):
    # Explicit projection for this practice task, not the official full CORD metric.
    gt = json.loads(text)["gt_parse"]
    menu = gt.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    def scalar(value):
        if value is not None and not isinstance(value, str):
            raise UnsupportedAnnotation(f"Non-scalar field: {type(value).__name__}")
        return value
    sub = gt.get("sub_total", {})
    total = gt.get("total", {})
    if not isinstance(sub, dict) or not isinstance(total, dict):
        raise UnsupportedAnnotation("Multiple receipt-level groups")
    result = {
        "items": [{"name": scalar(m.get("nm")), "quantity": scalar(m.get("cnt")),
                   "price": scalar(m.get("price"))} for m in menu],
        "subtotal": scalar(sub.get("subtotal_price")),
        "tax": scalar(sub.get("tax_price")),
        "total": scalar(total.get("total_price")),
    }
    if not validate_schema(result):
        raise ValueError("Invalid projected annotation")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/cord")
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    out = Path(args.output)
    if any(out.glob("*.jsonl")):
        raise SystemExit("Existing manifests found; choose a new output directory to preserve splits.")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        raise SystemExit("limit must be positive")
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET, revision=REVISION)
    counts, original_counts, all_hashes, excluded = {}, {}, {}, []
    for split in ("train", "validation", "test"):
        records = []
        images = out / "images" / split
        images.mkdir(parents=True, exist_ok=True)
        ds = dataset[split]
        if args.limit_per_split:
            ds = ds.select(range(min(args.limit_per_split, len(ds))))
        original_counts[split] = len(ds)
        print(f"Preparing {split}: {len(ds)} source images", flush=True)
        for index, row in enumerate(ds):
            if index and index % 100 == 0:
                print(f"  {split}: {index}/{len(ds)}", flush=True)
            image = row["image"].convert("RGB")
            # Preserve the dataset's split. Index is stable at the pinned revision.
            sample_id = f"{split}-{index:04d}"
            path = images / f"{sample_id}.png"
            image.save(path)
            try:
                target = target_from_ground_truth(row["ground_truth"])
            except UnsupportedAnnotation as exc:
                excluded.append({"id": sample_id, "split": split, "reason": str(exc),
                    "image_path": str(path.relative_to(out)),
                    "gt_parse": json.loads(row["ground_truth"])["gt_parse"]})
                continue
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            all_hashes.setdefault(digest, []).append(sample_id)
            records.append({"id": sample_id, "image_path": str(path.relative_to(out)),
                            "target": target, "width": image.width, "height": image.height,
                            "image_sha256": digest})
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        counts[split] = len(records)
        print(f"  {split}: {len(records)}/{len(ds)} eligible", flush=True)
    provenance = {"dataset": DATASET, "revision": REVISION, "counts": counts,
                  "original_counts": original_counts, "excluded_count": len(excluded),
                  "exclusion_policy": "Quarantine non-scalar annotations; do not guess a target. Scores apply only to eligible subset.",
                  "license": "CC-BY-4.0", "projection": "items(nm,cnt,price), sub_total(subtotal_price,tax_price), total(total_price)",
                  "duplicate_image_groups": [v for v in all_hashes.values() if len(v) > 1]}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "excluded.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+'\n' for r in excluded))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
