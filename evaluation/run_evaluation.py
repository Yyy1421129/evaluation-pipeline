import sys
import os
import json
import argparse
import ast
from tqdm import tqdm
from datetime import datetime

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

TASK_ALIASES = {
    "asr": ["asr"],
    "ser": ["ser", "emotion_recognition"],
    "gr": ["gr", "gender_recognition"],
    "s2tt": ["s2tt", "translation_ec"],
    "slu": ["slu", "stress_based_reasoning"],
    "sd": ["sd", "speaker_diarization"],
    "sa-asr": ["sa-asr", "sa_asr"]
}

def get_task_name(task_input):
    """
    Get standardized task name, supports case-insensitive and aliases
    """
    if not task_input:
        return None
    
    task_lower = task_input.lower()
    
    # Direct match
    if task_lower in TASK_MAP:
        return TASK_MAP[task_lower]
    
    # Alias match
    for canonical_name, aliases in TASK_ALIASES.items():
        if task_lower in [alias.lower() for alias in aliases]:
            return TASK_MAP[canonical_name]
    
    return None

def format_task_result(task_name, result, num_samples=None):
    """
    Format task result and add computed metrics
    """
    task_result = {
        "task_type": task_name,
        "result": result
    }
    
    if num_samples is not None:
        task_result["num_samples"] = num_samples
    
    # ASR task: compute WER percentage
    if task_name == "asr_wer" and isinstance(result, dict) and "all" in result:
        if result["all"] != 0:
            wer_percent = (result["sub"] + result["del"] + result["ins"]) / result["all"] * 100
        else:
            wer_percent = 0.0
        task_result["wer_percent"] = round(wer_percent, 2)
    
    # ASR codeswitch task: compute MER, WER, CER percentage
    elif task_name == "asr_wer" and isinstance(result, tuple) and len(result) == 3:
        mer_score, wer_score, cer_score = result
        task_result["mer_percent"] = round(mer_score * 100, 2)
        task_result["wer_percent"] = round(wer_score * 100, 2)
        task_result["cer_percent"] = round(cer_score * 100, 2)
    
    # SER task: compute accuracy percentage
    elif task_name == "ser_eval" and isinstance(result, (int, float)):
        task_result["accuracy_percent"] = round(result * 100, 2)
    
    # GR task: compute accuracy percentage
    elif task_name == "gr_eval" and isinstance(result, (int, float)):
        task_result["accuracy_percent"] = round(result * 100, 2)
    
    # SLU task: compute accuracy percentage
    elif task_name == "slu_eval" and isinstance(result, (int, float)):
        task_result["accuracy_percent"] = round(result * 100, 2)
    
    # S2TT task: compute BLEU and chrF2 scores
    elif task_name == "s2tt_eval" and isinstance(result, dict):
        if "bleu" in result:
            task_result["bleu_score"] = round(result["bleu"], 2)
        if "chrf" in result:
            task_result["chrf_score"] = round(result["chrf"], 2)
    
    # SD task: compute DER percentage
    elif task_name == "sd_eval" and isinstance(result, dict):
        if "der" in result:
            task_result["der_percent"] = round(result["der"] * 100, 2)
        if "num_sessions" in result:
            task_result["num_sessions"] = result["num_sessions"]
    
    # SA-ASR task: compute cpWER and DER percentage
    elif task_name == "sa_asr_eval" and isinstance(result, dict):
        if "cpwer" in result:
            task_result["cpwer_percent"] = round(result["cpwer"] * 100, 2)
        if "der" in result:
            task_result["der_percent"] = round(result["der"] * 100, 2)
        if "num_sessions" in result:
            task_result["num_sessions"] = result["num_sessions"]
    
    return task_result

def load_gt_by_task(gt_json_path):
    task_dict = {}
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Loading GT data", unit="lines"):
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
        lines = f.readlines()
    for line in tqdm(lines, desc="Loading prediction data", unit="lines"):
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
    parser.add_argument("--collar", type=float, default=0.5, help="Collar value for SA-ASR evaluation (default: 0.5)")
    parser.add_argument("--saved", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help="Save results to file (default: true)")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results (default: results)")

    args = parser.parse_args()

    gt_json = args.gt_json
    pred_txt = args.pred_txt
    language = args.language
    ser_mapping = None
    task = args.task.lower()
    collar = args.collar
    saved = args.saved
    save_dir = args.save_dir
    if args.ser_mapping:
        try:
            ser_mapping = ast.literal_eval(args.ser_mapping)
        except Exception:
            print("[Warning] ser_mapping parse failed, using default.")
            ser_mapping = None
    
    gr_mapping = None
    if args.gr_mapping:
        try:
            gr_mapping = ast.literal_eval(args.gr_mapping)
        except Exception:
            print("[Warning] gr_mapping parse failed, using default.")
            gr_mapping = None

    evaluator = Evaluator(CONFIG, language=language, ser_mapping=ser_mapping, gr_mapping=gr_mapping)
    
    # Collect all results
    all_results = {
        "evaluation_time": datetime.now().isoformat(),
        "ground_truth": gt_json,
        "prediction": pred_txt,
        "language": language,
        "tasks": {}
    }

    if task in ["sd", "sa-asr", "sd_eval", "sa_asr_eval"]:
        task_name = get_task_name(task)
        if not task_name:
            print(f"[Warning] Unknown task type: {task}, skip.")
            sys.exit(1)
        print(f"\n=== Evaluating Task: {task.upper()} (special input format) ===")
        data = {
            "ref_file": gt_json,
            "hyp_file": pred_txt,
            "collar": collar
        }
        result = evaluator.run(task_name, data, language)
        task_result = format_task_result(task_name, result)
        all_results["tasks"][task_name] = task_result
    else:
        task_dict = load_gt_by_task(gt_json)
        pred_dict = load_pred(pred_txt)
        for task, items in tqdm(task_dict.items(), desc="Processing tasks", unit="task"):
            task_name = get_task_name(task)
            if not task_name:
                print(f"[Warning] Unknown task type: {task}, skip.")
                continue
            print(f"\n=== Evaluating Task: {task.upper()} ===")
            ref_lines = []
            hyp_lines = []
            for item in tqdm(items, desc=f"Processing {task.upper()} items", unit="item", leave=False):
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
            result = evaluator.run(task_name, data, language)
            task_result = format_task_result(task_name, result, num_samples=len(items))
            all_results["tasks"][task_name] = task_result
            os.remove(ref_file)
            os.remove(hyp_file)
    
    # Save results
    if saved:
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        gt_basename = os.path.splitext(os.path.basename(gt_json))[0]
        pred_basename = os.path.splitext(os.path.basename(pred_txt))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{gt_basename}_{pred_basename}_{timestamp}.json"
        result_path = os.path.join(save_dir, result_filename)
        
        # Save results to JSON file
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] Results saved to: {result_path}")