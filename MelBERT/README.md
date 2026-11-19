# Tái lập MelBERT & RoBERTa Baseline trên VUA-18

Phần thực nghiệm mô hình **MelBERT** và mô hình đối chứng **RoBERTa-base** cho bài toán nhận diện ẩn dụ trong `NLP_PROJECT`.

## Nguồn tham khảo

- Paper: [MelBERT: Metaphor Detection via Contextualized Embeddings](https://aclanthology.org/2021.naacl-main.141/) (Choi et al., NAACL 2021).
- Mã nguồn tác giả: https://github.com/jin530/MelBERT
- Commit nguồn: `11b0f78dae93c71c56b805dc28d810f6050c86c4`.
- Tài liệu hướng dẫn gốc của tác giả lưu tại: `README_ORIGINAL.md`.

Mô hình và thuật toán MelBERT thuộc công trình của tác giả. Phần việc thực hiện gồm:
1. Chuẩn bị thực nghiệm và môi trường chạy trên Kaggle GPU Tesla T4.
2. Tái lập mô hình MelBERT (kết hợp biểu diễn ngữ cảnh và biểu diễn từ cô lập / MIPVU).
3. Huấn luyện mô hình đối chứng `RoBERTa-base` (vanilla fine-tuning) trên cùng tập dữ liệu và siêu tham số để so sánh hiệu quả của cơ chế biểu diễn đối lập trong MelBERT.
4. Xuất dự đoán và đánh giá chi tiết theo từ loại (POS: VERB, NOUN, ADJ, ADV).

## Dữ liệu thực nghiệm

- Sử dụng tập dữ liệu **VUA-18** (chuẩn split của tác giả):
  - **Train**: 101.975 tokens (12.8% ẩn dụ, 6.268 câu)
  - **Dev**: 34.253 tokens (13.1% ẩn dụ, 1.541 câu)
  - **Test**: 43.947 tokens (14.2% ẩn dụ, 2.679 câu)
- Không lưu dữ liệu dung lượng lớn và checkpoint mô hình trên GitHub; toàn bộ được quản lý trong môi trường Kaggle.

## Cấu hình huấn luyện

- **Backbone**: `roberta-base` (124.6M tham số).
- **Môi trường**: Kaggle GPU Tesla T4, PyTorch 2.x, Transformers.
- **Siêu tham số chính**:
  - Epochs: 3
  - Batch size: 32 (hoặc 16 tùy GPU mem)
  - Learning rate: 3e-5
  - Max sequence length: 150
  - Class weight: 3
  - Warmup epoch: 2
  - Drop ratio: 0.2
- **Tiêu chí chọn Checkpoint**: Chọn checkpoint có F1 cao nhất trên tập Validation/Dev, sau đó nạp để đánh giá một lần duy nhất trên tập Test.

## Kết quả thực nghiệm

### 1. Đối chiếu với Paper trên tập Test VUA-18 (All-POS)

| Mô hình | Precision | Recall | F1 | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **MelBERT (Paper công bố)** | 80.1% | 76.9% | **78.5%** | — |
| **MelBERT (Tái lập thực tế)** | **81.0%** | 75.9% | **78.3%** | **94.0%** |
| **RoBERTa-base (Baseline)** | 78.6% | **76.1%** | **77.4%** | 93.7% |

*Kết quả tái lập MelBERT đạt F1 **78.3%**, bám rất sát kết quả công bố trong bài báo gốc (78.5%), đồng thời đạt Precision ấn tượng lên tới **81.0%**.*

### 2. Chi tiết kết quả kiểm thử theo từ loại (POS Breakdown)

| POS Track | Số mẫu (n) | MelBERT F1 | MelBERT Acc | RoBERTa F1 | RoBERTa Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ALL-POS** | 43.947 | **78.3%** | **94.0%** | 77.4% | 93.7% |
| **VERB** | 9.247 | **75.0%** | **90.0%** | 74.6% | 89.4% |
| **NOUN** | 8.583 | **70.6%** | **91.8%** | 69.5% | 91.6% |
| **ADV** | 3.054 | **73.5%** | **96.0%** | 71.6% | 95.8% |
| **ADJ** | 3.965 | **66.3%** | **91.2%** | 64.4% | 90.8% |

## Cấu trúc thư mục

```text
MelBERT/
├── README.md               # Báo cáo thực nghiệm và hướng dẫn
├── README_ORIGINAL.md      # Tài liệu gốc của tác giả MelBERT
├── requirements.txt        # Các thư viện phụ thuộc
├── main.py                 # Script huấn luyện & đánh giá chính
├── modeling.py             # Định nghĩa kiến trúc MelBERT & các biến thể
├── data_loader.py          # Bộ tải và tiền xử lý dữ liệu
├── main_config.cfg         # File cấu hình tham số mặc định
├── utils/                  # Các hàm tiện ích
├── data_sample/            # Dữ liệu mẫu (sample) để chạy thử nghiệm
├── notebooks/
│   ├── melbert.ipynb       # Notebook Kaggle huấn luyện & đánh giá MelBERT
│   └── roberta.ipynb       # Notebook Kaggle huấn luyện baseline RoBERTa
└── results/
    ├── paper_comparison.csv # Bảng đối chiếu chỉ số với paper
    └── metrics.json        # Dữ liệu JSON chi tiết các metric trên dev/test
```
