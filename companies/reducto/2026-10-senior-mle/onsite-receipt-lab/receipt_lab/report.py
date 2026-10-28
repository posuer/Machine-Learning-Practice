"""Generate a local, static inspection artifact; escape model text as untrusted HTML."""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from .common import load_rows
from .evaluate import read_predictions, score_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/cord/validation.jsonl")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", default="runs/review.html")
    args = p.parse_args()
    rows = load_rows(args.manifest, args.limit)
    baseline = read_predictions(args.baseline)
    candidate = read_predictions(args.candidate) if args.candidate else {}
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    escape = lambda x: html.escape(str(x))
    content = ['<!doctype html><meta charset="utf-8"><title>Receipt review</title>',
               '<style>body{font-family:system-ui;margin:24px}section{border-top:1px solid #aaa;padding:20px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
               '<h1>Receipt extraction review</h1><p>Raw model output, ground truth and custom field F1. Read failures as well as successes.</p>']
    for row in rows:
        source = quote(os.path.relpath(row['image_path'], out.parent), safe='/')
        content.append(f'<section><h2>{escape(row["id"])}</h2><div class="grid"><div><img src="{source}"></div>')
        content.append('<div><h3>Ground truth</h3><pre>'+escape(json.dumps(row['target'],ensure_ascii=False,indent=2))+'</pre></div>')
        for name, mapping in [('Baseline', baseline), ('Candidate', candidate)]:
            prediction = mapping.get(row['id'], {})
            score = score_one(row['target'], prediction)
            content.append(f'<div><h3>{name}</h3><p>Field F1: {score["field_f1"]:.3f}</p><pre>{escape(prediction.get("raw_text","MISSING"))}</pre></div>')
        content.append('</div></section>')
    out.write_text('\n'.join(content))
    print(out)


if __name__ == '__main__':
    main()
