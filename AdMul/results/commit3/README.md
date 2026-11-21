# Kết quả huấn luyện AdMul trên VUA All
## Kết quả cuối

Huấn luyện hoàn tất 8 epoch, seed 4. Chọn **epoch 8 bằng F1 validation cao nhất**

| Độ đo | Validation (%) | Test (%) |
|---|---:|---:|
| Accuracy | 94.81 | 94.56 |
| Precision | 75.84 | 77.93 |
| Recall | 81.17 | 78.52 |
| F1 | 78.42 | 78.23 |

Giá trị đầy đủ chưa làm tròn nằm trong `history.json` và `result.json`. Đây là điểm nhận diện ẩn dụ VUA All (`md_dev`/`md_test`); điểm BSD-test là tác vụ phụ và được lưu riêng trong lịch sử.

## Tiến độ theo epoch

| Epoch | Dev F1 (%) | Test F1 (%) |
|---|---:|---:|
| 1 | 71.59 | 73.79 |
| 2 | 75.16 | 74.11 |
| 3 | 77.40 | 77.35 |
| 4 | 78.26 | 77.69 |
| 5 | 76.51 | 77.25 |
| 6 | 77.71 | 77.77 |
| 7 | 78.27 | 77.79 |
| 8 | 78.42 | 78.23 |

Log ghi nhận các phiên kết thúc sau epoch **1, 3, 5, 7, 8**; resume từ epoch 1, 3, 5, 7. Trường `session_epochs: 1` trong `result.json` chỉ nói phiên cuối chạy một epoch; tổng số epoch hoàn thành là 8.

Tổng `seconds` của 8 epoch khoảng **36,62 giờ**, gồm train và các lượt đánh giá theo runner. Con số này không gồm toàn bộ thời gian chuẩn bị môi trường, tải đầu vào, lưu checkpoint, phiên bị gián đoạn hoặc thời gian chờ giữa phiên.

## Môi trường và cấu hình ghi nhận

- GPU: **1 Tesla T4**, CUDA 11.6.
- Python 3.10.14; PyTorch 1.13.1+cu116; Transformers 4.18.0; NumPy 1.21.3.
- VUA All: train 116622 mẫu, dev 38628 mẫu, test 50175 mẫu.
- BSD train lấy 116622 mẫu đầu từ SemCor threshold 3; BSD-test 4081 mẫu.
- Batch 64, 8 epoch, LR `3e-5`, warmup 2 epoch, max length 150.
- Gradient checkpointing được bật; không mixed precision.

## Số tham số

Đếm từ metadata `epoch_08.pt` bằng script nhóm bổ sung `AdMul/scripts/count_parameters.py`,
không chạy lại huấn luyện. Không tính optimizer, scheduler hoặc buffer `position_ids`.

| Thành phần | Số tham số |
|---|---:|
| Backbone DeBERTa | 183.831.552 |
| Pooler | 590.592 |
| Classifier và discriminator | 7.690 |
| **Tổng** | **184.429.834** |

Số tham số được bổ sung sau khi tải Output; đây không phải trường đã được log trong phiên train.

## Các file trong mốc này

| File | Nội dung |
|---|---|
| `result.json` | Tóm tắt hoàn tất train và điểm test của epoch được chọn |
| `history.json` | Điểm, learning rate và thời gian từng epoch |
| `epoch_metrics.csv` | Bảng rút ra từ lịch sử, metric giữ thang 0–1 |
| `events.jsonl` | Các sự kiện lưu checkpoint, resume và kết thúc phiên |
| `console.log` | Log runner tích lũy qua các phiên |
| `kaggle_kernel.log` | Log notebook do Kaggle CLI xuất, phạm vi phiên notebook được tải |
| `identity.json` | Hash đầu vào, cấu hình và thông tin môi trường |
| `config.yaml`, `requirements.lock.txt` | Cấu hình thực tế và danh sách dependency |
| `admul_run.json` | Định dạng phiên chạy và commit nguồn tác giả |
| `source.json` | Nguồn lấy output và hash các file gốc |
| `verification.json` | Kết quả đối chiếu tính nhất quán giữa các file |


