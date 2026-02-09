import sys
import os
import json
import argparse

from evaluator import Evaluator
from config import CONFIG

TASK_MAP = {
    "asr": "asr_wer",
    "ser": "ser_eval",
    "gr": "gr_eval",
    "s2tt": "s2tt_eval",
    "slu": "slu_eval",
    "sd": "sd_eval",
    "sa-asr": "sa_asr_eval"
}

def load_gt_by_task(gt_json_path):
    task_dict = {}
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task = item['task'].lower()
            if task not in task_dict:
                task_dict[task] = []
            task_dict[task].append(item)
    return task_dict

def load_pred(pred_path):
    pred_dict = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                pred_dict[parts[0]] = parts[1]
    return pred_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation tasks")
    parser.add_argument("gt_json", help="Ground truth JSON file")
    parser.add_argument("pred_txt", help="Prediction TXT file")
    parser.add_argument("--language", default="en", help="Normalization language (default: en)")
    parser.add_argument("--ser_mapping", type=str, help="SER mapping dict, e.g. '{\"neu\":0,\"hap\":1,\"ang\":2,\"sad\":3}'")
    parser.add_argument("--gr_mapping", type=str, help="GR mapping dict, e.g. '{\"man\":0,\"woman\":1}'")
    parser.add_argument("--task", type=str, default="", help="Task name (sd or sa-asr for special format)")

    args = parser.parse_args()

    gt_json = args.gt_json
    pred_txt = args.pred_txt
    language = args.language
    ser_mapping = None
    task = args.task.lower()
    if args.ser_mapping:
        import ast
        try:
            ser_mapping = ast.literal_eval(args.ser_mapping)
        except Exception:
            print("[Warning] ser_mapping parse failed, using default.")
            ser_mapping = None
    
    gr_mapping = None
    if args.gr_mapping:
        import ast
        try:
            gr_mapping = ast.literal_eval(args.gr_mapping)
        except Exception:
            print("[Warning] gr_mapping parse failed, using default.")
            gr_mapping = None

    evaluator = Evaluator(CONFIG, language=language, ser_mapping=ser_mapping, gr_mapping=gr_mapping)

    if task in ["sd", "sa-asr"]:
        task_name = TASK_MAP.get(task, None)
        if not task_name:
            print(f"[Warning] Unknown task type: {task}, skip.")
            sys.exit(1)
        print(f"\n=== Evaluating Task: {task.upper()} (special input format) ===")
        data = {
            "ref_file": gt_json,
            "hyp_file": pred_txt
        }
        evaluator.run(task_name, data, language)
    else:
        task_dict = load_gt_by_task(gt_json)
        pred_dict = load_pred(pred_txt)
        for task, items in task_dict.items():
            task_name = TASK_MAP.get(task, None)
            if not task_name:
                print(f"[Warning] Unknown task type: {task}, skip.")
                continue
            print(f"\n=== Evaluating Task: {task.upper()} ===")
            ref_lines = []
            hyp_lines = []
            for item in items:
                key = item['key']
                ref = item['target']
                hyp = pred_dict.get(key, "")
                ref_lines.append(f"{key}\t{ref}")
                hyp_lines.append(f"{key}\t{hyp}")
            ref_file = f"tmp_ref_{task}.txt"
            hyp_file = f"tmp_hyp_{task}.txt"
            with open(ref_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(ref_lines) + '\n')
            with open(hyp_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hyp_lines) + '\n')
            data = {
                "ref_file": ref_file,
                "hyp_file": hyp_file,
                "case_sensitive": False,
                "tochar": False,
                "verbose": 1
            }
            if task == "slu":
                data["prompt_jsonl"] = gt_json
            evaluator.run(task_name, data, language)
            os.remove(ref_file)
            os.remove(hyp_file)