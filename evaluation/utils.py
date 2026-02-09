def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]