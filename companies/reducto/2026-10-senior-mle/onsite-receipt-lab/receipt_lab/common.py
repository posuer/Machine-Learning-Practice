import json
from pathlib import Path

PROMPT = '''Read this receipt image and return ONLY a JSON object with these keys:
"items": an array in printed receipt order, each object with "name", "quantity", "price";
"subtotal", "tax", "total": receipt-level values.
All values except items must be strings or null. Within items, all three values must
also be strings or null. Copy the visible text, preserving number separators,
punctuation and case. Use null when a field is not present. Do not calculate missing
amounts. price is the printed line price, not the unit price. quantity is the printed
quantity expression, including units if shown. Extract top-level purchased menu
lines; do not turn modifiers/submenu lines into separate items. No markdown fences,
explanations or extra keys.'''


def load_config(path):
    return json.loads(Path(path).read_text())


def load_rows(manifest, limit=None):
    path = Path(manifest).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("Manifest is empty")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    for row in rows:
        row["image_path"] = str((path.parent / row["image_path"]).resolve())
    return rows


def messages(answer=None):
    result = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}
    ]}]
    if answer is not None:
        result.append({"role": "assistant", "content": [
            {"type": "text", "text": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}
        ]})
    return result


def validate_schema(obj):
    keys = {"items", "subtotal", "tax", "total"}
    if not isinstance(obj, dict) or set(obj) != keys:
        return False
    scalar = lambda x: x is None or isinstance(x, str)
    return (all(scalar(obj[k]) for k in keys - {"items"})
            and isinstance(obj["items"], list)
            and all(isinstance(item, dict) and set(item) == {"name", "quantity", "price"}
                    and all(scalar(v) for v in item.values()) for item in obj["items"]))


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError(f"Non-JSON constant: {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=bad_constant)
