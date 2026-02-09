#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_prediction.py
根据 multitask.jsonl 中的 prompt 把字母答案还原成完整文本
输入：
  - multitask.jsonl           # 含 prompt 及 key
  - pred.txt(或自定义路径)  # 每行 key<TAB>原始回答
输出：
  - processed_predictions_filtered.txt  # key<TAB>完整选项文本
"""

import json
import os
import re
import sys
import argparse

# -------------------- 参数配置 --------------------
parser = argparse.ArgumentParser(description="Restore full text answers from letter predictions.")
parser.add_argument("multitask_jsonl_path", help="Path to multitask.jsonl")
parser.add_argument("predictions_path", help="Path to predictions file (key<TAB>answer)")
parser.add_argument("output_path", help="Output file path")
args = parser.parse_args()

multitask_jsonl_path = args.multitask_jsonl_path
predictions_path = args.predictions_path
output_path = args.output_path
# -------------------------------------------------

key2prompt = {}
with open(multitask_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        prompt = item.get("prompt", "")
        opts = {}
        if prompt:
            for ln in prompt.splitlines():
                m = re.match(r'^\s*([A-Da-d])\.\s*(.*)$', ln)
                if m:
                    opts[m.group(1).upper()] = m.group(2).strip()
        key2prompt[item["key"]] = (prompt, opts)

from collections import defaultdict
key2full = defaultdict(str)
with open(predictions_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        key, pred = line.split("\t", 1)
        key2full[key] += " " + pred.strip()

processed = []
for key, full_pred in key2full.items():
    if key not in key2prompt or not key2prompt[key][1]:  
        processed.append((key, full_pred.strip()))
        continue
    _, opts = key2prompt[key]

    m = re.search(r'\b([A-Da-d])\b', full_pred)
    if m and m.group(1).upper() in opts:
        processed.append((key, opts[m.group(1).upper()]))
        continue

    pred_norm = re.sub(r'\s+', ' ', full_pred.strip().lower())
    for txt in opts.values():
        if pred_norm == re.sub(r'\s+', ' ', txt.lower()):
            processed.append((key, txt))
            break
    else:
        processed.append((key, full_pred.strip()))

with open(output_path, "w", encoding="utf-8") as fout:
    for k, v in processed:
        fout.write(f"{k}\t{v}\n")

print(f"Done! {len(processed)} lines written to {output_path}")
