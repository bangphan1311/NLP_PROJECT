# Tái lập RoPPT trên VUA-18

- Nguồn: paper [Metaphor Detection with Effective Context Denoising](https://arxiv.org/abs/2302.05611) (Wang et al., 2023) và code chính thức https://github.com/MajiBear000/RoPPT
- Phạm vi Deadline 1: đọc paper, dựng môi trường, chuẩn bị dữ liệu VUA-18, chạy full train, chọn checkpoint bằng dev, đánh giá toàn bộ test.
- Kết quả tóm tắt: dev F1 **0.7437** (epoch 3); test all-POS **P 78.20 / R 72.69 / F1 75.34** (paper: P 80.0 / R 78.2 / F1 79.1).

## Dữ liệu

- Dùng split VUA-18 gốc của tác giả (đã tiền xử lý spaCy sẵn: `*_rospacy.json`), tải từ link trong README repo gốc (Google Drive, mục `VUA18`).
- Đặt vào `data/VUA18/` gồm: `train.tsv, val.tsv, test.tsv` và `train_rospacy.json, val_rospacy.json, test_rospacy.json`.
- Số mẫu: train 101,975 / val 34,253 / test 43,947 (đơn vị: target token sau khi unroll).
- Không chạy lại parser (dùng đúng file rospacy của tác giả; `parser.sh` chỉ cần nếu muốn parse lại từ TSV).

## Môi trường

- Python 3.7.9 (conda-forge; repo ghi 3.7.7 nhưng không có sẵn), torch 1.6.0+cpu, transformers 4.2.2, spaCy 3.5.2 + en_core_web_trf 3.5.0.
- Lưu ý khi cài (xem chi tiết trong `CHANGES_DEVIATIONS.md`):
  - `allennlp==2.10.1` trong requirements xung đột với `torch==1.6.0` và chỉ dùng cho nhánh ELMo (không dùng ở đây) → import được bọc `try/except`, không cài allennlp.
  - `sacremoses==0.0.45` (bản 0.0.53 build từ source cần Cython >= 3.1).
  - Cài spaCy model: `en_core_web_trf-3.5.0` (bản mới yêu cầu spaCy >= 3.7).

## Cấu hình chạy

Giữ nguyên cấu hình tác giả (`main_config.cfg`), chỉ ghi rõ các tham số dòng lệnh:

```
python main.py --data_dir data/VUA18 --task_name vua18 --dataset_name vua18 \
  --model_type MELBERT_GAT --class_weight 3 --bert_model roberta-base \
  --num_train_epoch 3 --train_batch_size 16 --learning_rate 3e-5 --warmup_epoch 2
```

- `train_batch_size 16` = mặc định `main_config.cfg` (dòng VUA-20 trong `scripts/run.sh` dùng 32; máy CPU 15GB RAM không đủ).
- Chọn checkpoint theo dev, sau đó nạp checkpoint tốt nhất và đánh giá toàn bộ test.

## Quá trình huấn luyện (đã chạy)

- Epoch 1–2: CPU (i7-1255U, 12 thread, torch 1.6.0+cpu). ~9h46m/epoch train + ~50 phút dev eval.
  - dev F1: epoch 1 = 0.6509, epoch 2 = 0.7194.
- Epoch 3: tiếp tục từ checkpoint epoch 2 trên Google Colab T4 (torch 2.x, transformers mới, patch import `AdamW`), `--num_train_epoch 1 --warmup_epoch 0`.
  - dev F1 = 0.7437 → checkpoint cuối.
- Test: P 78.20 / R 72.69 / F1 75.34 / Acc 93.24.
- Thời gian/ phần cứng/ số tham số: `results/run_record.json` (127.0M tham số).

Notebook Kaggle/Colab để chạy epoch cuối: `notebooks/RoPPT_colab.ipynb` (nhận `roppt_upload.zip` đặt ở My Drive).

## Kết quả

`results/`:

| File | Nội dung |
|---|---|
| `metrics_recomputed.json` | P/R/F1/Acc tính lại từ file dự đoán, all-POS và verb-only |
| `paper_comparison.csv` | So sánh với số công bố (all-POS) |
| `errors_30_sample.csv` | 15 FP + 15 FN (seed 42), có sentence_id/word_index/token |
| `predictions_test.tsv` | Dự đoán từng token test: `idx, pred, gold, target, sentence` |
| `experiments.log` | Log dev/test của lượt chạy cuối |
| `run_record.json` | Command, thời gian, phần cứng, số tham số, cấu hình |
| `error_analysis_RoPPT.md` | Bảng so sánh, kiểm tra file dự đoán, 15 FP + 15 FN |
| `config.json`, `main_config.cfg` | Cấu hình model và cấu hình chạy |

- File dự đoán: 43,947 dòng = đúng số target test, idx liên tục 0..43946, đã đối chiếu target với `test_rospacy.json`.
- Chênh lệch so với paper chủ yếu ở recall (72.7 vs 78.2): dev F1 vẫn tăng ở epoch cuối (0.651 → 0.719 → 0.744), tức chưa hội tụ so với lịch 3 epoch liên tục của tác giả (epoch 3 chạy lại lịch LR).

## Checkpoint

`pytorch_model.bin` (~485MB) không đưa lên git (quá giới hạn 100MB/file). Xin link tải hoặc chép trực tiếp từ máy chạy; file nằm trong `saves/roberta-base/3_20261118-0759/`.

## Khác biệt so với tác giả

Xem `CHANGES_DEVIATIONS.md`: chọn checkpoint theo dev + lưu checkpoint cho vua18, dump dự đoán test, sửa `cuda:1` → `cuda:0`, bỏ allennlp (nhánh ELMo không dùng), chia epoch 3 sang GPU. Không đổi dữ liệu, mô hình, loss hay cấu hình huấn luyện cốt lõi.

## Cách chạy lại

1. Cài môi trường theo mục **Môi trường** (hoặc `conda create -n roppt -c conda-forge python=3.7.9`; cài torch 1.6.0+cpu trước, rồi requirements trừ allennlp).
2. Tải dữ liệu VUA-18 vào `data/VUA18/` theo mục **Dữ liệu**.
3. Chạy lệnh ở mục **Cấu hình chạy**.
4. Kết quả ghi vào `saves/roberta-base/<run>/` (checkpoint, log, `predictions_test.tsv`).
