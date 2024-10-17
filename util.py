def read_tsv(file_path, has_label=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if has_label:
                data.append((parts[0], parts[1]))
            else:
                data.append(parts[0])
    return data
