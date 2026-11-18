# Deviations from the official RoPPT code/config (Bạn 4)

Team rule: stick to the paper + official code for data, preprocessing and config.
Every difference below is listed explicitly. None affect the model's math or data.

## Environment
1. Python 3.7.9 instead of 3.7.7 (3.7.7 unavailable in conda-forge; same 3.7 series).
2. `allennlp==2.10.1` omitted: its dependency spec conflicts with `torch==1.6.0`
   required by the same requirements.txt (allennlp 2.10.1 demands torch>=1.10).
   allennlp is only used for ELMo embeddings (`datasets.py`,
   `embedding_type == 'elmo'`); RoPPT runs with `embedding_type = melbert`,
   so this code path is never executed. Import made optional in `datasets.py`.
3. torch kept at the pinned `1.6.0+cpu` (CPU-only machine: Intel Iris Xe, no NVIDIA GPU).
4. `sacremoses==0.0.45` installed instead of letting pip pick: 0.0.53 has no wheel
   and its build requires Cython>=3.1 (Python>=3.8). Pure runtime no-op for this repo.
5. spaCy model `en_core_web_trf-3.5.0` (matches spaCy 3.5.2; latest model releases
   require spaCy>=3.7). Used only if re-parsing data; the authors' preprocessed
   `*_rospacy.json` files were downloaded and used as-is.

## Code (main.py / datasets.py)
6. `main.py`: device was hardcoded `cuda:1` -> changed to `cuda:0` (crashes on
   single-GPU machines). Irrelevant on this CPU-only box.
7. `main.py`: checkpoint selection now uses the **dev/val** set during training and
   saves `pytorch_model.bin` when dev F1 improves (original only saved when
   `task_name == 'vua'`, a name that doesn't exist in this repo's loaders, so
   vua18/vua20 checkpoints were never saved). Final test inference reloads the
   best-dev checkpoint first. This implements the team agreement: "chọn checkpoint
   bằng dev, đánh giá toàn bộ test".
8. `main.py`: final test predictions are dumped to
   `saves/roberta-base/<run>/predictions_test.tsv`
   (columns: idx, pred, gold, target, sentence; row order = unrolled dataset order)
   for the shared team evaluation script (P/R/F1 by Bạn 3).

## Run config for VUA-18 (paper config otherwise)
- `--task_name vua18 --data_dir data/VUA18 --model_type MELBERT_GAT
  --bert_model roberta-base --class_weight 3 --num_train_epoch 3
  --learning_rate 3e-5 --warmup_epoch 2`
- `train_batch_size 16` = `main_config.cfg` default. `scripts/run.sh`'s commented
  VUA-20 line uses 32; 32 OOMs this 16GB-RAM CPU machine.
- `eval_batch_size 4` (cfg default). Seed 42 (cfg default).

## Final VUA-18 run (split CPU/GPU)
- Epochs 1-2 on local CPU (torch 1.6.0+cpu, batch 16, warmup 2 as configured)
- Epoch 3 resumed from the epoch-2 checkpoint on Google Colab T4 GPU:
  torch 2.x (Colab preinstalled) + transformers latest (AdamW import patched to torch.optim
  in main.py and trainer.py), `--num_train_epoch 1 --warmup_epoch 0` (linear decay only,
  approximating the tail of the original 3-epoch schedule)
- Channel stays in [0,1] EMA teacher updates (unchanged); no changes to data, model or loss
- Final: dev F1 0.7437, test P 0.7820 / R 0.7269 / F1 0.7534 (paper: P 80.0 / R 78.2 / F1 79.1)
