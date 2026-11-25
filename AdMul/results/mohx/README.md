# Tái lập AdMul trên MOH-X và so sánh với paper

Thực nghiệm đã hoàn tất **10/10 fold**, mỗi fold 5 epoch, seed 4. Trung bình F1 theo fold là **86,60%** khi chọn epoch có held-out F1 cao nhất, hoặc **84,45%** khi dùng epoch cuối cố định. Paper công bố **87,90%**; hai mức chênh lệch tương ứng là **−1,30** và **−3,45 điểm phần trăm**.

## Nguồn và phạm vi

- Paper: Shenglong Zhang và Ying Liu (2023), [Adversarial Multi-task Learning for End-to-end Metaphor Detection](https://aclanthology.org/2023.findings-acl.96/).
- Số liệu so sánh: [Bảng 4, trang 1489](https://aclanthology.org/2023.findings-acl.96.pdf#page=7); cấu hình tham khảo: [Bảng 3, trang 1488](https://aclanthology.org/2023.findings-acl.96.pdf#page=6).
- Mã tác giả: [SilasTHU/AdMul](https://github.com/SilasTHU/AdMul), commit `4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8`.
- Output thực nghiệm: [ngtk61000/moh-x trên Kaggle](https://www.kaggle.com/code/ngtk61000/moh-x), tải ngày 25/11/2026. Ngày tải không phải ngày huấn luyện; chưa ghi nhận số phiên bản Output.

Đây là thí nghiệm huấn luyện MOH-X theo cross-validation, đối chiếu với Bảng 4. Cấu trúc mô hình và thuật toán thuộc tác giả; nhóm bổ sung phần chạy Kaggle, quản lý tiến độ và đối chiếu số liệu.

## Dữ liệu và giao thức đánh giá

MOH-X có **647 target**, gồm 315 nhãn ẩn dụ và 332 nhãn nghĩa đen theo dữ liệu dự đoán đã kiểm tra. Runner shuffle một lần với seed 4 rồi chia thành 10 đoạn liên tiếp theo `main_kf.py`: fold 1–9 có 64 target kiểm tra và 583 target train; fold 10 có 71 target kiểm tra và 576 target train. Đây không phải cách chia stratified.

Mỗi fold khởi tạo từ pretrained và huấn luyện 5 epoch. [result.json](result.json) ghi hai cách tổng hợp:

1. **`author_best_heldout_fold_mean`**: chọn epoch có F1 cao nhất trên chính held-out fold, rồi lấy trung bình từng metric của 10 fold. Cách này tái hiện lựa chọn trong code tác giả, **không phải chọn bằng tập dev độc lập**; kết quả có thể lạc quan do dùng nhãn held-out để chọn epoch.
2. **`final_epoch_fold_mean`**: lấy epoch 5 cho mọi fold, rồi tính trung bình từng metric. Cách này không chọn epoch bằng held-out fold, nhưng khác quy tắc chọn kết quả ở cách thứ nhất.

Các chỉ số P/R/F1 tính cho lớp ẩn dụ (`label=1`), không phải macro-F1 giữa hai lớp. JSON lưu thang 0–1; các bảng trình bày theo %. F1 trung bình theo fold không phải F1 tính từ precision và recall trung bình.

## So sánh với paper

| Kết quả | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) |
|---|---:|---:|---:|---:|
| AdMul trong paper, Bảng 4 | 87,40 | 88,80 | 87,90 | 88,00 |
| **Nhóm — epoch tốt nhất theo held-out, trung bình 10 fold** | **88,37** | **85,33** | **86,60** | **87,34** |
| Nhóm − paper, điểm phần trăm | +0,97 | −3,47 | **−1,30** | −0,66 |
| **Nhóm — epoch cuối cố định, trung bình 10 fold** | **89,68** | **80,32** | **84,45** | **85,78** |
| Nhóm − paper, điểm phần trăm | +2,28 | −8,48 | **−3,45** | −2,22 |

Chênh lệch được tính từ các giá trị đã làm tròn hai chữ số. Precision của cả hai cách báo điểm cao hơn số công bố, nhưng recall thấp hơn. Mức giảm recall lớn hơn mức tăng precision, đi kèm F1 thấp hơn paper.

Chọn epoch theo held-out làm F1 trung bình cao hơn epoch cuối **2,15 điểm**. Cần giữ rõ quy tắc này khi sử dụng mức 86,60%; không diễn giải nó là kết quả trên test hoàn toàn độc lập với quá trình chọn checkpoint. Các fold và seed của paper chưa được đối chiếu trực tiếp với lần chạy này nên bảng trên là đối chiếu số công bố, chưa phải đánh giá ghép cặp trên cùng phân hoạch.

## Kết quả từng fold

| Fold | Số target held-out | Epoch tốt nhất theo held-out | F1 epoch tốt nhất (%) | F1 epoch 5 (%) |
|---|---:|---:|---:|---:|
| 1 | 64 | 3 | 86,67 | 82,14 |
| 2 | 64 | 4 | 82,76 | 81,36 |
| 3 | 64 | 4 | 87,88 | 83,87 |
| 4 | 64 | 4 | 81,63 | 80,77 |
| 5 | 64 | 3 | 87,18 | 83,58 |
| 6 | 64 | 5 | 88,14 | 88,14 |
| 7 | 64 | 4 | 87,27 | 82,35 |
| 8 | 64 | 2 | 91,67 | 89,55 |
| 9 | 64 | 5 | 86,27 | 86,27 |
| 10 | 71 | 4 | 86,49 | 86,49 |
| **Trung bình 10 fold** | **647 tổng cộng** | — | **86,60** | **84,45** |

Epoch tốt nhất thay đổi từ 2 đến 5. Các fold có tập kiểm tra nhỏ nên khác biệt giữa fold không thay thế đánh giá độ ổn định qua nhiều seed.

## Kiểm tra dự đoán gộp của epoch cuối

Đã kiểm tra `predictions_oof_final.csv` trong Output có **647 row ID khác nhau**, mỗi target xuất hiện một lần, và tính lại confusion matrix khớp `final_epoch_pooled` trong [result.json](result.json):

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 253 | 30 | 62 | 302 |

| Cách tính | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) |
|---|---:|---:|---:|---:|
| Gộp 647 dự đoán ở epoch 5 | 89,40 | 80,32 | 84,62 | 85,78 |

F1 gộp **84,62%** khác trung bình F1 theo fold **84,45%** do cách tổng hợp. Không thay hai số này cho nhau khi so sánh. Với dự đoán epoch cuối, FN=62 lớn hơn FP=30, phù hợp với recall thấp hơn precision.

## Cấu hình và chi phí thực nghiệm

| Hạng mục | Giá trị ghi nhận |
|---|---|
| Backbone | `microsoft/deberta-v3-base` |
| Fold / epoch mỗi fold / seed | 10 / 5 / 4 |
| Batch / learning rate / max length | 32 / `2e-5` / 150 |
| BSD / dropout | SemCor threshold 8 / 0 |
| Trọng số BSD / global / local | 0,2 / 0,1 / 0,1 |
| Số tham số mỗi model | 184.429.834 |
| GPU | 1 Tesla T4 |
| Python / CUDA | 3.10.14 / 11.6 |
| PyTorch / Transformers / NumPy | 1.13.1+cu116 / 4.18.0 / 1.21.3 |
| Gradient checkpointing / mixed precision | Bật / tắt |
| Tổng thời gian train | **70,81 phút** |
| Tổng thời gian train + đánh giá | **86,42 phút** |
| Peak GPU memory allocated lớn nhất giữa các fold | **4,20 GiB** |

Thời gian lấy từ `train_seconds` và `train_and_eval_seconds`; chưa bao gồm đầy đủ chuẩn bị môi trường, tokenization, lưu checkpoint hoặc chờ giữa phiên. Peak memory là giá trị PyTorch ghi nhận cho tensor được cấp phát, không phải toàn bộ VRAM tiến trình. Xem [config.yaml](config.yaml), [identity.json](identity.json) và các bản ghi fold trong [result.json](result.json).

## Minh chứng và giới hạn

Thiết lập được tái lập theo code và cấu hình tác giả công bố, chưa xác nhận giống toàn bộ môi trường tạo ra điểm paper. Riêng BSD, phần mô tả trong paper nêu threshold 3, còn [YAML MOH-X của tác giả](https://github.com/SilasTHU/AdMul/blob/4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8/configs/mohx.yaml) đặt threshold 8; lần chạy này dùng 8 theo YAML. Gradient checkpointing là phần bật thêm để giảm VRAM. Seed 4 khớp mặc định code, nhưng chưa xác nhận là seed tạo ra số liệu trong paper.

- [result.json](result.json): đủ 10 fold, hai cách chọn epoch, metric từng fold, điểm gộp và thời gian.
- [config.yaml](config.yaml), [identity.json](identity.json): cấu hình, runtime, hash dữ liệu và mã nguồn.
- [source.json](source.json): nguồn Kaggle, SHA-256 các tệp minh chứng và dự đoán dùng để kiểm tra.

Chỉ có một seed và một phân hoạch 10 fold; chưa có kiểm định ý nghĩa thống kê hoặc đánh giá nhiều seed. Paper ghi RTX 3090, lần tái lập dùng Tesla T4; chưa đủ bằng chứng để quy khoảng cách metric cho phần cứng hay yếu tố riêng nào. Báo cáo chưa phân tích lỗi theo lemma/ngữ cảnh.

Dữ liệu, pretrained, model từng fold và dự đoán đầy đủ giữ ở Output Kaggle và bản tải cục bộ `moh-x/AdMul_MOHX/`. Thư mục báo cáo lưu các minh chứng nhỏ để có thể đưa lên GitHub.
