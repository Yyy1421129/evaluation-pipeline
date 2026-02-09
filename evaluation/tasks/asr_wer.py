import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wenet_compute_cer import Calculator, characterize, normalize

def compute_wer(ref_file, hyp_file, ignore_words=None, case_sensitive=False, tochar=False, split=None, verbose=1):
    calculator = Calculator()
    rec_set = {}

    with open(hyp_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            if tochar:
                array = characterize(line)
            else:
                array = line.strip().split()
            if len(array) == 0: continue
            fid = array[0]
            rec_set[fid] = normalize(array[1:], ignore_words or set(), case_sensitive, split)

    results = []
    for line in open(ref_file, 'r', encoding='utf-8'):
        if tochar:
            array = characterize(line)
        else:
            array = line.rstrip('\n').split()
        if len(array) == 0: continue
        fid = array[0]
        if fid not in rec_set:
            continue
        lab = normalize(array[1:], ignore_words or set(), case_sensitive, split)
        rec = rec_set[fid]
        result = calculator.calculate(lab, rec)
        results.append(result)
        if verbose:
            if result['all'] != 0:
                wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
            else:
                wer = 0.0
            print(f'utt: {fid}')
            print(f'WER: {wer:.2f} % N={result["all"]} C={result["cor"]} S={result["sub"]} D={result["del"]} I={result["ins"]}')
    overall = calculator.overall()
    if overall['all'] != 0:
        wer = float(overall['ins'] + overall['sub'] + overall['del']) * 100.0 / overall['all']
    else:
        wer = 0.0
    print(f'Overall -> {wer:.2f} % N={overall["all"]} C={overall["cor"]} S={overall["sub"]} D={overall["del"]} I={overall["ins"]}')
    return overall