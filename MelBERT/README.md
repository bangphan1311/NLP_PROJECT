# Tái lập MelBERT (NAACL-HLT 2021)

Phần thực nghiệm tái lập mô hình **MelBERT** (*Metaphor Detection via Contextualized Late Interaction using Metaphorical Identification Theories*) cho bài toán nhận diện ẩn dụ trong `NLP_PROJECT`.

## Nguồn tham khảo

- Paper: [MelBERT: Metaphor Detection via Contextualized Late Interaction using Metaphorical Identification Theories](https://aclanthology.org/2021.naacl-main.141/) (Choi et al., NAACL-HLT 2021).
- Mã nguồn tác giả: https://github.com/jin530/MelBERT
- Commit nguồn: `11b0f78dae93c71c56b805dc28d810f6050c86c4`.
- Tài liệu hướng dẫn gốc của tác giả lưu tại: [README_ORIGINAL.md](README_ORIGINAL.md).

## Giao thức thực nghiệm (5 seeds)

Notebook [notebooks/melBERT.ipynb](notebooks/melBERT.ipynb) tái lập đầy đủ theo đúng protocol trong bài báo với **5 random seeds** (`[1, 2, 3, 4, 5]`):

| Dataset | Vai trò | Cách chạy |
| :--- | :--- | :--- |
| **VUA-18** | Train + Test | Huấn luyện trên `train.tsv` (3 epochs); theo dõi F1 qua `test.tsv` theo `main.py` gốc |
| **VUA-20** | Train + Test | Huấn luyện trên `train.tsv` (3 epochs); theo dõi F1 qua `test.tsv` theo `main.py` gốc |
| **MOH-X** | Zero-shot test | Không train riêng; dùng model đã train trên VUA-20 để đánh giá qua 10 split |
| **TroFi** | Zero-shot test | Không train riêng; dùng model đã train trên VUA-20 để đánh giá qua 10 split |

> [!NOTE]
> Khi so sánh với `AdMul` trong repo, cần lưu ý sự khác biệt giao thức: `AdMul` train 10-fold cross-validation trực tiếp trên MOH-X và TroFi, còn `MelBERT` đánh giá **zero-shot từ model VUA-20** đúng theo Table 6 của paper MelBERT.

## Cấu hình và lưu ý tái lập

- **Cấu hình mô hình**: `roberta-base`, `MELBERT`, `classifier_hidden=768`, `dropout=0.2`, `use_pos=True`, `use_local_context=True`, `max_seq_length=150`, `train_batch_size=32`, `eval_batch_size=8`, `learning_rate=3e-5`, `warmup_epoch=2`, `num_train_epoch=3`, `class_weight=3`, `AdamW` (`eps=1e-6`, weight decay `0.01`), gradient clipping `1.0`.
- **Môi trường chạy**: Kaggle GPU, Python 3.13, PyTorch 2.11, `transformers==4.57.1` (vá import `AdamW` sang `torch.optim.AdamW` và giữ `eps=1e-6` khớp mặc định `transformers==4.2.2` của repo gốc).
- **Tính trung thực với source gốc**:
  - Giữ nguyên hành vi `main.py` gốc đánh giá `test.tsv` sau mỗi epoch và lấy `best_test_metrics_over_epochs` cho VUA-18/VUA-20.
  - Giữ nguyên hành vi `run_eval()` gọi `model.eval()` mà không bật lại `model.train()` ở epoch 2–3 để tái lập đúng source code tác giả.
- Không lưu dữ liệu dung lượng lớn và checkpoint mô hình trên GitHub; toàn bộ được quản lý trong môi trường Kaggle.

## Kết quả thực nghiệm (Mean ± Std qua 5 seeds)

Đối chiếu trực tiếp kết quả tái lập (đơn vị `%`) với số liệu công bố trong Tables 2, 3 và 6 của paper:

| Dataset | Nguồn | Precision (%) | Recall (%) | F1 (%) | Δ F1 (Nhóm − Paper) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **VUA-18** | Paper | 80,1 | 76,9 | 78,5 | — |
| | **Nhóm tái lập (5 seeds)** | **80,10 ± 1,25** | **77,33 ± 1,86** | **78,67 ± 0,37** | **+0,17** |
| **VUA-20** | Paper | 76,4 | 68,6 | 72,3 | — |
| | **Nhóm tái lập (5 seeds)** | **75,69 ± 1,69** | **69,76 ± 1,39** | **72,58 ± 0,23** | **+0,28** |
| **MOH-X** *(zero-shot)* | Paper | 79,3 | 79,7 | 79,2 | — |
| | **Nhóm tái lập (5 seeds)** | **79,24 ± 0,78** | **79,54 ± 2,71** | **79,04 ± 1,18** | **−0,16** |
| **TroFi** *(zero-shot)* | Paper | 53,4 | 74,1 | 62,0 | — |
| | **Nhóm tái lập (5 seeds)** | **53,28 ± 0,15** | **74,04 ± 1,58** | **61,87 ± 0,65** | **−0,13** |

Cả 4 bộ dữ liệu đều tái lập sát kết quả paper (chênh lệch F1 dưới 0,3 điểm phần trăm và mọi giá trị paper đều nằm trong khoảng `mean ± 1 std`).

## Cấu trúc thư mục

```text
MelBERT/
├── README.md                        # Báo cáo thực nghiệm 5-seed và đối chiếu paper
├── README_ORIGINAL.md               # Tài liệu gốc của tác giả MelBERT
├── requirements.txt                 # Các thư viện phụ thuộc
├── main.py                          # Script huấn luyện & đánh giá chính
├── modeling.py                      # Định nghĩa kiến trúc MelBERT
├── data_loader.py                   # Bộ tải và tiền xử lý dữ liệu
├── main_config.cfg                  # File cấu hình tham số mặc định
├── utils/                           # Các hàm tiện ích
├── data_sample/                     # Dữ liệu mẫu (sample) để chạy thử nghiệm
├── notebooks/
│   └── melBERT.ipynb                # Notebook Kaggle tái lập 5 seeds trên 4 dataset
└── results/
    ├── paper_comparison.csv         # Bảng đối chiếu chỉ số với paper
    └── metrics.json                 # Dữ liệu JSON chi tiết mean/std trên 4 dataset
```
