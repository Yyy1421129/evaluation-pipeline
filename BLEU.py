#!/usr/bin/env python3
# compute_bleu.py
import sys
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF


def load_lines_plain(file_path):
    """读取纯文本格式，每行一句，去除空行"""
    return [ln.strip() for ln in open(file_path, encoding='utf-8') if ln.strip()]

def main(ref, pred, lang="zh"):
    refs = load_lines_plain(ref)
    preds = load_lines_plain(pred)

    print(f'Loaded {len(refs)} refs, {len(preds)} preds')
    if not refs or not preds:
        print('Error: refs or preds is empty!')
        sys.exit(1)
    if len(refs) != len(preds):
        print(f'Warning: refs and preds line count mismatch!')

    if lang.lower() in ["zh", "ch", "chinese"]:
        bleu = BLEU(tokenize='zh')
        chrf = CHRF(word_order=2)
        bleu_name = "BLEU (中文分词)"
    elif lang.lower() in ["en", "english"]:
        bleu = BLEU(tokenize='13a')
        chrf = CHRF(word_order=2)
        bleu_name = "BLEU (英文分词)"
    else:
        bleu = BLEU(tokenize='none')
        chrf = CHRF(word_order=2)
        bleu_name = f"BLEU (tokenize=none, lang={lang})"

    score_bleu = bleu.corpus_score(preds, [refs])
    print(f'{bleu_name} = {score_bleu.score:.2f}')

    bleu_char = BLEU(tokenize='char')
    score_char = bleu_char.corpus_score(preds, [refs])
    print(f'BLEU (字符级别) = {score_char.score:.2f}')

    chrf_score = chrf.corpus_score(preds, [refs])
    print(f'chrF2 = {chrf_score.score:.2f}')

if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        print('用法: python compute_bleu.py ref.txt pred.txt [lang]')
        sys.exit(1)
    ref = sys.argv[1]
    pred = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) == 4 else "zh"
    main(ref, pred, lang)
