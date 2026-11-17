# Thiết lập thực nghiệm AdMul

## Dữ liệu

Chạy tác vụ `all` với `configs/vua_all.yaml`. Mỗi dòng dữ liệu là một mẫu có từ đích trong ngữ cảnh, không phải một câu duy nhất trong corpus.

| Vai trò | File dưới thư mục dữ liệu | Số mẫu dự kiến |
|---|---|---:|
| Train nhận diện ẩn dụ | `VUA_All/train.tsv` | 116622 |
| Validation nhận diện ẩn dụ | `VUA_All/val.tsv` | 38628 |
| Test nhận diện ẩn dụ | `VUA_All/test.tsv` | 50175 |
| Train tác vụ phụ BSD | `semcor/threshold/train_3.tsv` | 130808 trước khi cắt |
| Test BSD | `semcor/threshold/test_3.tsv` | 4081 |

Theo `main_vua.py` của tác giả, lấy 116622 mẫu đầu của SemCor train rồi shuffle để ghép với VUA train. Validation và test không được ghép vào dữ liệu train.

Nguồn tải dữ liệu được tác giả cung cấp trong [README gốc](https://github.com/SilasTHU/AdMul/blob/4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8/README.md). Kiểm tra file cục bộ tại `results/commit1/data_check.json`; kiểm tra này xác nhận số dòng, nhãn và hash, chưa chứng minh không có trùng lặp giữa các split.

## Mô hình và cấu hình

- Backbone: `microsoft/deberta-v3-base`.
- Cấu hình hiệu lực: mặc định từ `configs/default.py`, ghi đè bằng `configs/vua_all.yaml`.
- Seed: 4; số epoch: 8; train/eval batch size: 64.
- Learning rate: `3e-5`; weight decay: `0.01`; warmup: 2 epoch.
- Max length: 150; dropout: 0.0; trọng số lớp ẩn dụ: 4; ngưỡng SemCor: 3.
- Train đồng thời nhận diện ẩn dụ và tác vụ phụ BSD theo code tác giả.

Lệnh gốc khi đã có đủ mã nguồn, dependency và model pretrained:

```bash
python main_vua.py --task all --cfg ./configs/vua_all.yaml
```

Lệnh này là tham chiếu tới repo tác giả; commit chuẩn bị hiện tại chỉ chứa tài liệu và cấu hình. Khi chạy cần đặt `DATA.data_dir` trỏ đến thư mục chứa `VUA_All/` và `semcor/`, và `DATA.plm` trỏ đến model pretrained.

## Đánh giá và so sánh

- Ghi accuracy, precision, recall và F1. Đối chiếu định nghĩa lớp dương và phạm vi từ được chấm trước khi so sánh với RNN_HG.
- Chọn checkpoint bằng F1 validation; báo cáo test của checkpoint được chọn.
- Code gốc đánh giá cả validation, MD-test và BSD-test mỗi epoch.
- Trước khi so sánh, kiểm tra cùng dữ liệu/split và cùng cách tính metric; số mẫu bằng nhau chưa đủ chứng minh dữ liệu giống nhau.
- Ghi nhận phần cứng, phiên bản thư viện, thời gian train và số tham số ở mốc chạy thực tế.

README tác giả yêu cầu Transformers 4.18.0 và NumPy 1.21.3. Thiết lập Kaggle, các điều chỉnh môi trường và checkpoint/resume sẽ được ghi riêng trong mốc tiếp theo.
