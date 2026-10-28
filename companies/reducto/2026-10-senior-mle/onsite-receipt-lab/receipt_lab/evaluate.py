"""Strict projected-schema evaluation. This is not the official CORD benchmark scorer."""
import argparse
import json
import math
from pathlib import Path

from .common import load_rows, strict_json, validate_schema


def normalize(value):
    # Only collapse whitespace. Do not erase decimal signs, punctuation or case.
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    return value


def leaves(value, prefix=""):
    if isinstance(value, dict):
        return set().union(*(leaves(v, f"{prefix}.{k}") for k, v in value.items())) if value else set()
    if isinstance(value, list):
        return set().union(*(leaves(v, f"{prefix}[{i}]") for i, v in enumerate(value))) if value else set()
    return {(prefix, normalize(value))} if value is not None else set()


def score_one(target, prediction):
    try:
        parsed = strict_json(prediction.get("raw_text", ""))
        json_ok = True
    except (ValueError, TypeError):
        parsed, json_ok = None, False
    schema_ok = json_ok and validate_schema(parsed)
    usable = schema_ok and not prediction.get("truncated", False) and prediction.get("status") != "error"
    gold = leaves(target)
    guessed = leaves(parsed) if usable else set()
    tp, fp, fn = len(gold & guessed), len(guessed - gold), len(gold - guessed)
    return {"json_valid": json_ok, "schema_valid": schema_ok, "usable": usable,
            "exact": usable and normalize(parsed) == normalize(target),
            "tp": tp, "fp": fp, "fn": fn,
            "field_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float(usable),
            "total_correct": usable and normalize(parsed["total"]) == normalize(target["total"])}


def summarize(rows, predictions):
    ids = {row["id"] for row in rows}
    if set(predictions) - ids:
        raise ValueError("Predictions contain IDs outside the selected manifest")
    scored = [(row, score_one(row["target"], predictions.get(row["id"], {}))) for row in rows]
    n = len(rows)
    if not n:
        raise ValueError("No evaluation samples")
    tp, fp, fn = (sum(s[k] for _, s in scored) for k in ("tp", "fp", "fn"))
    times = sorted(float(p["latency_seconds"]) for p in predictions.values()
                   if isinstance(p.get("latency_seconds"), (int, float))
                   and math.isfinite(p["latency_seconds"]) and p["latency_seconds"] >= 0)
    def quantile(q):
        return times[max(0, math.ceil(q*len(times))-1)] if times else None
    totals = [s for r, s in scored if r["target"]["total"] is not None]
    result = {"samples": n, "predictions_present": len(predictions), "missing": n-len(predictions),
              "json_valid_rate": sum(s["json_valid"] for _, s in scored)/n,
              "schema_valid_rate": sum(s["schema_valid"] for _, s in scored)/n,
              "document_exact_match": sum(s["exact"] for _, s in scored)/n,
              "field_micro_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else sum(s["usable"] for _,s in scored)/n,
              "total_exact_match_on_present_labels": sum(s["total_correct"] for s in totals)/len(totals) if totals else None,
              "total_label_count": len(totals),
              "needs_review_or_failure_rate": sum(not s["usable"] for _,s in scored)/n,
              "truncated_count": sum(bool(p.get("truncated")) for p in predictions.values()),
              "latency": {"measured_samples": len(times), "p50": quantile(.5), "p95": quantile(.95), "p99": quantile(.99),
                          "scope": "preprocess+generate+decode; first request includes warmup; missing samples have no latency",
                          "tail_estimate_warning": "p99 on 100 or fewer samples is unstable; not a production SLO estimate"},
              "slices": {}}
    for name, predicate in {
        "long_receipt": lambda r: len(r["target"]["items"]) >= 8,
        "short_receipt": lambda r: len(r["target"]["items"]) < 8,
        "tall_image": lambda r: r["height"] / r["width"] >= 2.5,
    }.items():
        selected = [s for r,s in scored if predicate(r)]
        result["slices"][name] = {"n": len(selected), "mean_field_f1": sum(s["field_f1"] for s in selected)/len(selected) if selected else None}
    return result


def read_predictions(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["id"] in result:
            raise ValueError("Duplicate prediction ID")
        result[row["id"]] = row
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/cord/validation.jsonl")
    p.add_argument("--predictions", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    metrics = summarize(load_rows(args.manifest, args.limit), read_predictions(args.predictions))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
