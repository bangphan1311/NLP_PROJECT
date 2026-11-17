"""Count samples/labels and hash source TSV files without loading a model."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = {
        'VUA_All/train.tsv': 116622,
        'VUA_All/val.tsv': 38628,
        'VUA_All/test.tsv': 50175,
        'semcor/threshold/train_3.tsv': 130808,
        'semcor/threshold/test_3.tsv': 4081,
    }
    result = {
        'scope': 'Local row counts, label distributions and SHA-256; no split overlap check',
        'files': {},
    }
    for name, count in expected.items():
        path = args.data_root / name
        labels = Counter()
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream, delimiter='\t')
            next(reader)
            for row in reader:
                labels[row[-1]] += 1
        samples = sum(labels.values())
        if samples != count:
            raise ValueError(f'{name}: expected {count}, got {samples}')
        result['files'][name] = {
            'samples': samples,
            'label_counts': dict(sorted(labels.items())),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Checked {len(expected)} files; saved {args.output}')


if __name__ == '__main__':
    main()
