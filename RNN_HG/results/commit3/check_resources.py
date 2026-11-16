
import ast
import csv
import json
import sys
from pathlib import Path
import h5py
import numpy as np

repo = Path(sys.argv[1])
logs = Path(sys.argv[2])

# Đọc từ vựng train
vocab = set()
train_path = repo / "data/VUAsequence/VUA_seq_formatted_train.csv"

with train_path.open(encoding="latin-1", newline="") as f:
    for row in csv.DictReader(f):
        vocab.update(row["sentence"].split())

# Kiểm tra vector GloVe
matched = set()
glove_path = repo / "glove/glove840B300d.txt"

with glove_path.open(encoding="utf-8") as f:
    for line in f:
        fields = line.rstrip().split()

        if len(fields) != 301 or fields[0] not in vocab:
            continue

        vector = np.asarray(fields[1:], dtype=np.float32)
        assert vector.shape == (300,)
        assert np.isfinite(vector).all()
        matched.add(fields[0])

assert matched, "Khong tim thay GloVe cho tu vung train."

report = {
    "glove": {
        "dimension": 300,
        "train_vocabulary": len(vocab),
        "matched_words": len(matched),
        "unmatched_words": len(vocab - matched)
    },
    "vua": {}
}

# Kiểm tra mọi câu VUA
for split in ("train", "val", "test"):
    data_path = repo / "data/VUAsequence" / (
        "VUA_seq_formatted_" + split + ".csv"
    )
    elmo_path = repo / "elmo" / ("VUA_" + split + ".hdf5")

    sentences = tokens = metaphors = 0

    with h5py.File(str(elmo_path), "r") as embeddings:
        with data_path.open(encoding="latin-1", newline="") as f:
            for row in csv.DictReader(f):
                sentence = row["sentence"]
                words = sentence.split()
                labels = ast.literal_eval(row["label_seq"])
                pos = ast.literal_eval(row["pos_seq"])

                assert len(words) == len(labels) == len(pos)
                assert set(labels) <= {0, 1}
                assert sentence in embeddings, (
                    "Missing ELMo", split, row["txt_id"], row["sen_ix"]
                )
                assert embeddings[sentence].shape == (len(words), 1024), (
                    "Wrong ELMo shape", split, row["txt_id"], row["sen_ix"]
                )

                sentences += 1
                tokens += len(words)
                metaphors += sum(labels)

    report["vua"][split] = {
        "sentences": sentences,
        "tokens": tokens,
        "metaphor_tokens": metaphors,
        "elmo_dimension": 1024
    }

# Đối chiếu đầy đủ dữ liệu
expected = {
    "train": (6323, 116622, 13051),
    "val": (1550, 38628, 4487),
    "test": (2694, 50175, 6243)
}

for split, counts in expected.items():
    result = report["vua"][split]
    assert (
        result["sentences"],
        result["tokens"],
        result["metaphor_tokens"]
    ) == counts

(logs / "resource_check.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print(json.dumps(report, indent=2))
print("FULL_RESOURCE_OK")
