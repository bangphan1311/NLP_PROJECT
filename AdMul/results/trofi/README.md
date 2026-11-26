# Tái lập AdMul trên TroFi và so sánh với bài báo

Đã hoàn tất **10 fold × 10 epoch, seed 4**. F1 trung bình theo fold đạt **73,08%** khi chọn epoch tốt nhất theo held-out, thấp hơn AdMul trong paper (**74,70%**) **1,62 điểm phần trăm**. Khi dùng epoch 10 cố định, F1 đạt **69,58%**, thấp hơn paper **5,12 điểm**.

## Nguồn và phạm vi

- Paper: Shenglong Zhang và Ying Liu (2023), [Adversarial Multi-task Learning for End-to-end Metaphor Detection](https://aclanthology.org/2023.findings-acl.96/), Findings of ACL.
- Kết quả tham chiếu: [Bảng 4, trang 1489](https://aclanthology.org/2023.findings-acl.96.pdf#page=7); cấu hình: [Bảng 3, trang 1488](https://aclanthology.org/2023.findings-acl.96.pdf#page=6).
- Mã nguồn tác giả: [SilasTHU/AdMul](https://github.com/SilasTHU/AdMul), commit `4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8` theo metadata Output.
- Output: [lminhhinlminhhin/trofi trên Kaggle](https://www.kaggle.com/code/lminhhinlminhhin/trofi), tải ngày **26/11/2026**, múi giờ Asia/Saigon. Ngày tải không phải ngày train; chưa ghi nhận số phiên bản Output.

Đây là huấn luyện trên TroFi theo cross-validation, so với Bảng 4; không phải thí nghiệm zero-shot Bảng 8. Kiến trúc và thuật toán thuộc tác giả; nhóm bổ sung runner Kaggle, lưu/resume, xuất dự đoán và kiểm tra kết quả. Chưa khẳng định tái hiện đúng mọi điều kiện của lần chạy tạo ra số công bố.

## Dữ liệu và cách đánh giá

TroFi có **3.737 target**, gồm **1.627 nhãn ẩn dụ (43,54%)** và **2.110 nhãn nghĩa đen**, xác nhận từ dự đoán ngoài fold. Runner shuffle một lần với seed 4 rồi chia 10 đoạn liên tiếp theo `main_kf.py`, không stratified: fold 1–9 có 373 target held-out / 3.364 target train; fold 10 có 380 held-out / 3.357 train. Mỗi fold dùng số mẫu BSD train bằng số target MD train và khởi tạo model từ pretrained.

[result.json](result.json) lưu hai quy tắc báo điểm:

1. **`author_best_heldout_fold_mean`**: chọn epoch có F1 cao nhất trên chính held-out fold, rồi lấy trung bình không trọng số của từng metric qua 10 fold. Quy tắc này có trong code tác giả `main_kf.py`, không dùng dev độc lập; điểm có thể lạc quan do dùng nhãn held-out để chọn checkpoint.
2. **`final_epoch_fold_mean`**: dùng epoch 10 cho mọi fold, rồi tính trung bình từng metric. Không chọn epoch bằng held-out, nhưng khác quy tắc thứ nhất.

P/R/F1 tính cho **lớp ẩn dụ (`label=1`)**, không phải macro-F1 hai lớp. JSON dùng thang 0–1, bảng dùng %. F1 trung bình theo fold không phải F1 tính từ precision và recall trung bình. Fold mean và metric gộp cũng khác nhau.

## So sánh với paper

| Kết quả | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) |
|---|---:|---:|---:|---:|
| AdMul trong paper, Bảng 4 | 70,50 | 79,80 | 74,70 | 76,50 |
| **Nhóm — epoch tốt nhất theo held-out, trung bình 10 fold** | **69,23** | **77,97** | **73,08** | **75,01** |
| Nhóm − paper, điểm phần trăm | −1,27 | −1,83 | **−1,62** | −1,49 |
| **Nhóm — epoch 10 cố định, trung bình 10 fold** | **70,84** | **68,86** | **69,58** | **73,83** |
| Nhóm − paper, điểm phần trăm | +0,34 | −10,94 | **−5,12** | −2,67 |

Chênh lệch tính từ các giá trị đã làm tròn hai chữ số. Theo quy tắc chọn epoch của code tác giả, cả precision và recall đều thấp hơn số công bố, với mức giảm recall lớn hơn. Kết quả epoch cuối có precision gần paper nhưng recall thấp hơn nhiều, kéo F1 xuống. Chọn epoch bằng held-out làm F1 trung bình cao hơn **3,50 điểm** so với epoch cuối. Không diễn giải mức 73,08% là đánh giá trên test độc lập với lựa chọn checkpoint.

Các phân hoạch và seed của lần chạy trong paper chưa được đối chiếu trực tiếp, nên đây là so sánh với số công bố, chưa phải đối chứng ghép cặp trên cùng fold. Chỉ có một seed; chưa có kiểm định ý nghĩa thống kê.

## Kết quả từng fold

| Fold | Target held-out | Epoch tốt nhất theo held-out | F1 epoch tốt nhất (%) | F1 epoch 10 (%) |
|---|---:|---:|---:|---:|
| 1 | 373 | 9 | 75,63 | 71,75 |
| 2 | 373 | 7 | 75,58 | 71,90 |
| 3 | 373 | 5 | 65,51 | 60,62 |
| 4 | 373 | 3 | 74,03 | 70,70 |
| 5 | 373 | 8 | 72,24 | 67,94 |
| 6 | 373 | 7 | 71,98 | 70,74 |
| 7 | 373 | 9 | 74,72 | 69,97 |
| 8 | 373 | 5 | 72,36 | 70,41 |
| 9 | 373 | 3 | 75,54 | 70,40 |
| 10 | 380 | 7 | 73,20 | 71,34 |
| **Trung bình 10 fold** | **3.737 tổng cộng** | — | **73,08** | **69,58** |

Nguồn: [fold_metrics.csv](fold_metrics.csv) và [result.json](result.json). Epoch tốt nhất thay đổi từ 3 đến 9. Fold 3 có F1 thấp nhất theo cả hai cách báo điểm; chưa có phân tích định tính để kết luận nguyên nhân. Khác biệt giữa fold không thay thế đánh giá qua nhiều seed.

## Kiểm tra dự đoán ngoài fold

Đã tính lại từ [predictions_oof_final.csv](predictions_oof_final.csv): đủ **3.737 row ID khác nhau**, mỗi target xuất hiện đúng một lần và dùng epoch 10. Dự đoán gộp khớp dự đoán epoch cuối của từng fold và `final_epoch_pooled` trong JSON.

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 1.119 | 470 | 508 | 1.640 |

| Cách tổng hợp epoch cuối | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) |
|---|---:|---:|---:|---:|
| Gộp 3.737 dự đoán ngoài fold | 70,42 | 68,78 | 69,59 | 73,83 |
| Trung bình metric 10 fold | 70,84 | 68,86 | 69,58 | 73,83 |

Hai hàng khác cách tổng hợp; không thay thế cho nhau khi trích dẫn. Dự đoán epoch cuối có FN=508 lớn hơn FP=470, phù hợp recall thấp hơn precision. Đây là thống kê lỗi nhị phân, chưa giải thích loại lỗi ngôn ngữ.

## Cấu hình và chi phí thực nghiệm

| Hạng mục | Giá trị ghi nhận |
|---|---|
| Backbone | DeBERTa-v3-base |
| Fold / epoch mỗi fold / seed | 10 / 10 / 4 |
| Batch / learning rate / max length | 64 / `1e-5` / 150 |
| BSD / dropout | SemCor threshold 8 / 0 |
| Trọng số BSD / global / local | 0,2 / 0,1 / 0,1 |
| Khoảng lambda GRL / weight decay | 0,1–0,8 / 0,01 |
| Số tham số mỗi model | 184.429.834, đều trainable theo log |
| GPU | 1 Tesla T4 |
| Python / CUDA | 3.10.14 / 11.6 |
| PyTorch / Transformers / NumPy | 1.13.1+cu116 / 4.18.0 / 1.21.3 |
| Gradient checkpointing / mixed precision | Bật / tắt (FP32) |
| Tổng thời gian train | **48.010,19 giây ≈ 13,34 giờ** |
| Tổng thời gian train + đánh giá | **50.144,80 giây ≈ 13,93 giờ** |
| Peak GPU memory allocated lớn nhất giữa các fold | **5,64 GiB** |

Epoch, batch và LR khớp cấu hình TroFi trong Bảng 3. Thời gian lấy từ tổng các bản ghi fold trong [result.json](result.json), không bao gồm đầy đủ chuẩn bị môi trường, tokenization, lưu checkpoint hoặc chờ giữa các phiên. Peak memory là lượng tensor PyTorch cấp phát, không phải toàn bộ VRAM tiến trình. Paper dùng RTX 3090; chưa có đối chứng để so sánh tốc độ hoặc quy chênh lệch F1 cho GPU. Xem [config.yaml](config.yaml), [identity.json](identity.json) và [requirements.lock.txt](requirements.lock.txt).

## Minh chứng và giới hạn

- [verification.json](verification.json): kiểm tra hoàn tất 10 fold, đủ 10 epoch/fold, phân hoạch train/held-out không giao nhau và phủ đủ dữ liệu, metric tính lại từ dự đoán khớp JSON, trung bình metric và tổng thời gian khớp, hash dữ liệu khớp metadata.
- [source.json](source.json): nguồn Output, ngày tải và SHA-256 của 36 file minh chứng được sao chép nguyên byte; [folds.json](folds.json) lưu phân hoạch cụ thể.
- [paper_comparison.csv](paper_comparison.csv) và [fold_metrics.csv](fold_metrics.csv): bảng số liệu để đối chiếu.
- Mỗi thư mục `fold_01/` đến `fold_10/` có `history.json`, `predictions_author_best.csv` và `predictions_final.csv`, phục vụ kiểm tra hai quy tắc báo điểm.

Checkpoint khoảng 704 MB mỗi fold và pretrained được giữ trong Output Kaggle/bản tải cục bộ, không đưa vào báo cáo GitHub. Lần chạy này chưa có nhiều seed, kiểm định thống kê hay thí nghiệm loại bỏ thành phần để giải thích khoảng cách với paper. Chưa thể quy khoảng cách cho nhiễu dữ liệu, phần cứng hoặc một thành phần mô hình cụ thể chỉ từ các bảng điểm trên.
