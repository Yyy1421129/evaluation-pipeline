import os
import json
import unicodedata

def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

def remove_punctuation(text):
    # Remove all Unicode punctuation
    return ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))

def normalize_target(text):
    if any(is_chinese(ch) for ch in text):
        # Chinese: remove punctuation only
        return remove_punctuation(text)
    else:
        # English: uppercase and remove punctuation
        return remove_punctuation(text).upper()

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            if 'target' in data:
                data['target'] = normalize_target(data['target'])
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

def find_jsonl_files(root_dir):
    jsonl_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.jsonl') and not fname.endswith('.normalized.jsonl'):
                jsonl_files.append(os.path.join(dirpath, fname))
    return jsonl_files

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PerceptionFront-EndSystemTesting')
    jsonl_files = find_jsonl_files(base_dir)
    print(f"Found {len(jsonl_files)} jsonl files in PerceptionFront-EndSystemTesting.")
    for fpath in jsonl_files:
        outpath = fpath[:-6] + '.normalized.jsonl'
        print(f"Normalizing: {fpath} -> {outpath}")
        process_file(fpath, outpath)
    print("All files processed.")
