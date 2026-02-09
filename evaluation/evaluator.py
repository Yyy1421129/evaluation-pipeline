import os
from preprocess import Preprocessor
from tasks.asr_wer import compute_wer
from clean_marks import strip_all_punct
class Evaluator:
    def __init__(self, config, language="en", ser_mapping=None, gr_mapping=None):
        self.config = config
        self.preprocessor = Preprocessor(lang=language)  
        self.ser_mapping = ser_mapping or {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
        self.gr_mapping = gr_mapping or {"man": 0, "woman": 1}

    def _normalize_label(self, label):
        label = label.strip().lower()
        synonyms = {
            # 情感
            "happy": "hap",
            "happiness": "hap",
            "neutral": "neu",
            "angry": "ang",
            "anger": "ang",
            "sadness": "sad",
            "sad": "sad",
            # 性别
            "male": "man",
            "m": "man",
            "man": "man",
            "female": "woman",
            "f": "woman",
            "woman": "woman"
        }
        return synonyms.get(label, label)
    
    def run(self, task_name, data, language="en"):
        if task_name == "asr_wer":
            ref_norm_file = "tmp_ref_norm.txt"
            hyp_norm_file = "tmp_hyp_norm.txt"
            ref_lines = []
            with open(data["ref_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, text = parts
                        text_norm = self.preprocessor.normalize(text)
                        ref_lines.append(f"{key}\t{text_norm}")
            hyp_lines = []
            with open(data["hyp_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, text = parts
                        text_norm = self.preprocessor.normalize(text)
                        hyp_lines.append(f"{key}\t{text_norm}")
            with open(ref_norm_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(ref_lines) + '\n')
            with open(hyp_norm_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hyp_lines) + '\n')

            strip_all_punct(ref_norm_file)
            strip_all_punct(hyp_norm_file)
            result = compute_wer(ref_norm_file, hyp_norm_file)
            os.remove(ref_norm_file)
            os.remove(hyp_norm_file)
            return result
        elif task_name == "ser_eval":
            ref_labels = []
            hyp_labels = []
            with open(data["ref_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, label = parts
                        norm_label = self._normalize_label(label)
                        ref_labels.append(norm_label)
            with open(data["hyp_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, label = parts
                        norm_label = self._normalize_label(label)
                        mapped = None
                        if norm_label.isdigit():
                            for k, v in self.ser_mapping.items():
                                if str(v) == norm_label:
                                    mapped = k
                                    break
                        else:
                            mapped = norm_label
                        hyp_labels.append(mapped)
            valid = [(r, h) for r, h in zip(ref_labels, hyp_labels) if r is not None and h is not None]
            if not valid:
                print("[SER] No valid labels for accuracy calculation.")
                return None
            correct = sum(1 for r, h in valid if r == h)
            acc = correct / len(valid)
            print(f"[SER] Accuracy: {acc:.4f} ({correct}/{len(valid)})")
            return acc
        elif task_name == "gr_eval":
            ref_labels = []
            hyp_labels = []
            with open(data["ref_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, label = parts
                        norm_label = self._normalize_label(label)
                        ref_labels.append(norm_label)
            with open(data["hyp_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, label = parts
                        norm_label = self._normalize_label(label)
                        mapped = None
                        if norm_label.isdigit():
                            for k, v in self.gr_mapping.items():
                                if str(v) == norm_label:
                                    mapped = k
                                    break
                        else:
                            mapped = norm_label
                        hyp_labels.append(mapped)
            valid = [(r, h) for r, h in zip(ref_labels, hyp_labels) if r is not None and h is not None]
            if not valid:
                print("[GR] No valid labels for accuracy calculation.")
                return None
            correct = sum(1 for r, h in valid if r == h)
            acc = correct / len(valid)
            print(f"[GR] Accuracy: {acc:.4f} ({correct}/{len(valid)})")
            return acc
        elif task_name == "s2tt_eval":
            ref_lines = []
            hyp_lines = []
            with open(data["ref_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, text = parts
                        ref_lines.append(text)
            with open(data["hyp_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, text = parts
                        hyp_lines.append(text)
            ref_txt = "tmp_ref_s2tt_bleu.txt"
            hyp_txt = "tmp_hyp_s2tt_bleu.txt"
            with open(ref_txt, 'w', encoding='utf-8') as f:
                f.write('\n'.join(ref_lines) + '\n')
            with open(hyp_txt, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hyp_lines) + '\n')
            import subprocess
            bleu_cmd = ["python", "BLEU.py", ref_txt, hyp_txt, language]
            print(f"[S2TT] Running BLEU/chrF2 evaluation: {' '.join(bleu_cmd)}")
            subprocess.run(bleu_cmd)

            os.remove(ref_txt)
            os.remove(hyp_txt)
        elif task_name == "slu_eval":
            import subprocess
            hyp_processed = "tmp_hyp_slu_processed.txt"
            prompt_jsonl = data.get("prompt_jsonl")
            subprocess.run([
                "python", "process_prediction.py",
                prompt_jsonl,
                data["hyp_file"],
                hyp_processed
            ])
            ref_answers = {}
            with open(data["ref_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, ans = parts
                        ref_answers[key] = ans.strip().lower()
            hyp_answers = {}
            with open(hyp_processed, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, ans = parts
                        hyp_answers[key] = ans.strip().lower()
            total = 0
            correct = 0
            for key in ref_answers:
                if key in hyp_answers:
                    total += 1
                    if ref_answers[key] == hyp_answers[key]:
                        correct += 1
            if total == 0:
                print("[SLU] No valid pairs for accuracy calculation.")
                return None
            acc = correct / total
            print(f"[SLU] Accuracy: {acc:.4f} ({correct}/{total})")
            os.remove(hyp_processed)
            return acc
        elif task_name == "sd_eval":
            ref_rttm = data["ref_file"]
            hyp_rttm = data["hyp_file"]
            import subprocess
            der_cmd = [
                "meeteval-der", "dscore",
                "-r", ref_rttm,
                "-h", hyp_rttm,
                "--collar", "0.25"
            ]
            print(f"[SD] Running DER evaluation: {' '.join(der_cmd)}")
            subprocess.run(der_cmd)
        elif task_name == "sa_asr_eval":
            import meeteval
            ref_stm = data["ref_file"]
            hyp_stm = data["hyp_file"]
            ref_norm_stm = "tmp_ref_sa_asr_norm.stm"
            hyp_norm_stm = "tmp_hyp_sa_asr_norm.stm"

            with open(ref_stm, 'r', encoding='utf-8') as fin, open(ref_norm_stm, 'w', encoding='utf-8') as fout:
                for line in fin:
                    parts = line.strip().split(maxsplit=5)
                    if len(parts) == 6:
                        norm_trans = self.preprocessor.normalize(parts[5])
                        parts[5] = norm_trans
                        fout.write(' '.join(parts) + '\n')
            with open(hyp_stm, 'r', encoding='utf-8') as fin, open(hyp_norm_stm, 'w', encoding='utf-8') as fout:
                for line in fin:
                    parts = line.strip().split(maxsplit=5)
                    if len(parts) == 6:
                        norm_trans = self.preprocessor.normalize(parts[5])
                        parts[5] = norm_trans
                        fout.write(' '.join(parts) + '\n')

            ref = meeteval.io.load(ref_norm_stm)
            hyp = meeteval.io.load(hyp_norm_stm)
            result = meeteval.wer.cpwer(ref, hyp)
            avg = meeteval.wer.combine_error_rates(result.values())
            print(f"[SA-ASR] cpWER: {avg.error_rate:.4f} (errors: {avg.errors}, length: {avg.length})")

            os.remove(ref_norm_stm)
            os.remove(hyp_norm_stm)
        else:
            print(f"[Warning] Unknown task: {task_name}")