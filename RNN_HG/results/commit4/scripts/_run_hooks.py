
import json
import os
import sys
import time
from pathlib import Path
import torch

OUT = Path(os.environ["RNN_HG_OUTPUT"])
OUT.mkdir(parents=True, exist_ok=True)
records = []
started = 0.0

def epoch_start():
    global started
    torch.cuda.synchronize()
    started = time.perf_counter()

def epoch_end(g, phase, epoch):
    torch.cuda.synchronize()
    record = {
        "phase": phase,
        "epoch": epoch + 1,
        "num_iter": g["num_iter"],
        "lr": g["rnn_optimizer"].param_groups[0]["lr"],
        "seconds": round(time.perf_counter() - started, 4)
    }
    records.append(record)

    with (OUT / "epoch_log.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")

    print("EPOCH_DONE", record, flush=True)

def save_final(g):
    model = g["RNNseq_model"]
    config = {
        "num_classes": model.output_to_label.out_features,
        "embedding_dim": model.rnn.input_size,
        "hidden_size": model.rnn.hidden_size,
        "num_layers": model.rnn.num_layers,
        "bidir": model.rnn.bidirectional,
        "dropout1": model.dropout_on_input_to_LSTM.p,
        "dropout2": model.rnn.dropout,
        "dropout3": model.dropout_on_input_to_linear_layer.p
    }

    metadata = {
        "seed": g["r"],
        "batch_size": g["batch_size"],
        "num_epochs": g["num_epochs"],
        "additional_epochs": g["additional_epochs"],
        "num_iter": g["num_iter"],
        "train_sentences": len(g["train_dataset_vua"]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "training_with_dev_seconds": sum(r["seconds"] for r in records),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_selection": "final_after_29_plus_8"
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "glove_state_dict": g["glove_embeddings"].state_dict(),
        "word2idx": g["word2idx"],
        "idx2word": g["idx2word"],
        "pos2idx": g["pos2idx"],
        "idx2pos": g["idx2pos"],
        "model_config": config,
        "metadata": metadata
    }

    torch.save(checkpoint, str(OUT / "rnn_hg_final_seed4.pt"))
    (OUT / "training_summary.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print("FINAL_CHECKPOINT_SAVED", flush=True)
