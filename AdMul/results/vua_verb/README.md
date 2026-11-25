# Tái lập AdMul trên VUA Verb và so sánh với paper

Thực nghiệm đã hoàn tất **5 epoch**, seed 4. Checkpoint được chọn bằng F1 validation cao nhất là **epoch 3**, có dev F1 **76,48%** và test F1 **75,65%**. So với AdMul trong paper, test F1 thấp hơn **2,65 điểm phần trăm**.

## Nguồn và phạm vi

- Paper: Shenglong Zhang và Ying Liu (2023), [Adversarial Multi-task Learning for End-to-end Metaphor Detection](https://aclanthology.org/2023.findings-acl.96/).
- Số liệu so sánh: [Bảng 4, trang 1489](https://aclanthology.org/2023.findings-acl.96.pdf#page=7); cấu hình tham khảo: [Bảng 3, trang 1488](https://aclanthology.org/2023.findings-acl.96.pdf#page=6).
- Mã tác giả: [SilasTHU/AdMul](https://github.com/SilasTHU/AdMul), commit `4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8`.
- Output thực nghiệm: [dd7thjun/vua-verb trên Kaggle](https://www.kaggle.com/code/dd7thjun/vua-verb), tải ngày 25/11/2026. Ngày tải không phải ngày huấn luyện; chưa ghi nhận số phiên bản Output.

Đây là benchmark **VUA Verb với một lượt huấn luyện riêng**, không phải hàng VERB trong phân tích POS của VUA All. Cấu trúc mô hình và thuật toán thuộc công trình tác giả; phần tái lập của nhóm gồm notebook, quản lý checkpoint và đối chiếu kết quả.

## Dữ liệu và cách chọn kết quả

| Split | Số target được gán nhãn |
|---|---:|
| Train | 15.516 |
| Validation | 1.724 |
| Test | 5.873 |

Số lượng trên được kiểm tra từ ba TSV trong Output. Đơn vị là target token, không phải số câu. Các chỉ số P/R/F1 tính cho lớp ẩn dụ (`label=1`); metric gốc trong JSON dùng thang 0–1, bảng dưới chuyển sang %.

Chỉ dùng **dev F1** để chọn checkpoint `epochs/epoch_03.pt`. Runner vẫn đánh giá test ở mỗi epoch để giữ thứ tự thực thi của lần train, nhưng không dùng test để chọn epoch. Đã kiểm tra `best_epoch_test` trong [result.json](result.json) khớp `md_test` của epoch 3 trong [history.json](history.json).

## So sánh kết quả test

| Kết quả | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) |
|---|---:|---:|---:|---:|
| **Nhóm tái lập, epoch 3 chọn theo dev** | **77,95** | **73,48** | **75,65** | **85,82** |
| AdMul trong paper, Bảng 4 | 78,50 | 78,10 | 78,30 | 87,00 |
| Nhóm − paper, điểm phần trăm | −0,55 | −4,62 | **−2,65** | −1,18 |

Chênh lệch được tính từ các giá trị phần trăm đã làm tròn hai chữ số. Recall giảm nhiều hơn precision: mô hình tái lập thu hồi được ít trường hợp ẩn dụ hơn, trong khi độ chính xác của các dự đoán dương gần mức công bố hơn. Bảng điểm tổng hợp chưa xác định được kiểu ngôn ngữ hay nguyên nhân cụ thể gây ra các trường hợp bỏ sót.

## Tiến độ theo epoch

| Epoch | Dev F1 (%) | Test F1 (%) |
|---|---:|---:|
| 1 | 72,61 | 72,27 |
| 2 | 73,36 | 74,04 |
| **3 — được chọn** | **76,48** | **75,65** |
| 4 | 72,14 | 70,82 |
| 5 | 75,32 | 74,26 |

Dev F1 giảm sau epoch 3 và không vượt lại mức này ở epoch 5. Vì vậy, điểm báo cáo lấy checkpoint theo dev, thay vì mặc định lấy epoch cuối. Việc epoch 3 cũng có test F1 cao nhất trong lần chạy này không thay đổi quy tắc chọn checkpoint.

## Cấu hình và chi phí thực nghiệm

| Hạng mục | Giá trị ghi nhận |
|---|---|
| Backbone | `microsoft/deberta-v3-base` |
| Epoch / batch / learning rate | 5 / 64 / `3e-5` |
| Seed / max length / dropout | 4 / 150 / 0 |
| BSD | SemCor, threshold 8 |
| Trọng số BSD / global / local | 0,2 / 0,1 / 0,1 |
| GPU | 1 Tesla T4 |
| Python / CUDA | 3.10.14 / 11.6 |
| PyTorch / Transformers / NumPy | 1.13.1+cu116 / 4.18.0 / 1.21.3 |
| Gradient checkpointing / mixed precision | Bật / tắt |
| Thời gian 5 epoch | **3,19 giờ**, tổng trường `seconds` trong lịch sử |

Learning rate được log cố định ở `3e-5` trong cả năm epoch. Thời gian trên gồm train và các lượt đánh giá, không phải thời gian train thuần; không bao gồm đầy đủ cài đặt môi trường, chuẩn bị dữ liệu và lưu checkpoint. Cấu hình và môi trường nằm trong [config.yaml](config.yaml) và [identity.json](identity.json).

## Minh chứng và giới hạn

Thiết lập được tái lập theo code và cấu hình tác giả công bố, chưa xác nhận giống toàn bộ môi trường tạo ra điểm paper. Riêng BSD, phần mô tả trong paper nêu threshold 3, còn [YAML VUA Verb của tác giả](https://github.com/SilasTHU/AdMul/blob/4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8/configs/vua_verb.yaml) đặt threshold 8; lần chạy này dùng 8 theo YAML. Gradient checkpointing là phần bật thêm để giảm VRAM. Seed 4 khớp mặc định code, nhưng chưa xác nhận là seed tạo ra số liệu trong paper.

- [result.json](result.json): trạng thái hoàn tất, epoch được chọn và metric test.
- [history.json](history.json): dev/test/BSD-test và thời gian từng epoch.
- [source.json](source.json): nguồn Kaggle và SHA-256 của các tệp minh chứng được sao chép nguyên trạng.
- [config.yaml](config.yaml), [identity.json](identity.json): cấu hình thực tế, runtime và hash đầu vào.

Đây là một seed, chưa có đánh giá độ ổn định qua nhiều lần chạy hoặc kiểm định ý nghĩa thống kê. Paper ghi RTX 3090, còn lần tái lập dùng Tesla T4; chưa có thực nghiệm kiểm soát để quy chênh lệch metric cho GPU, thư viện hay yếu tố riêng nào. Chưa có phân tích dự đoán từng target cho VUA Verb trong báo cáo này.

Checkpoint và dữ liệu lớn giữ ở Output Kaggle và bản tải cục bộ `vua-verb/AdMul_VUA_Verb/`; thư mục báo cáo chỉ lưu minh chứng nhỏ để có thể đưa lên GitHub.
