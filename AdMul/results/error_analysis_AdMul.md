# AdMul — VUA All results & error analysis

## Test metrics (best-dev checkpoint, full test set)

| Kết quả | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| **Nhóm tái lập** | 77.93 | 78.52 | **78.23** | 94.56 |
| AdMul (paper, VUA All) | 78.4 | 79.5 | 79.0 | 94.7 |
| Chênh lệch (điểm phần trăm) | -0.47 | -0.98 | -0.77 | -0.14 |

Đơn vị: %. Precision, Recall và F1 tính cho lớp ẩn dụ (`label=1`). Nguồn paper: [Bảng 4, trang 1489](https://aclanthology.org/2023.findings-acl.96.pdf).

Huấn luyện đủ 8 epoch, seed 4; chọn **epoch 8 theo dev F1 cao nhất: 78.42%**. Backbone DeBERTa-v3-base, batch 64, learning rate 3e-5, max length 150, GPU Tesla T4. Runner có đánh giá test mỗi epoch, nhưng tiêu chí chọn checkpoint chỉ dùng dev F1.

F1 thấp hơn paper khoảng **0.77 điểm phần trăm**, với mức giảm recall lớn hơn precision. Chưa có thực nghiệm đối chứng hoặc nhiều seed để xác định nguyên nhân chênh lệch.

## Confusion matrix

Test có **50,175 target token**: 43,932 không ẩn dụ và 6,243 ẩn dụ.

| Nhãn thật | Dự đoán không ẩn dụ (0) | Dự đoán ẩn dụ (1) |
|---|---:|---:|
| Không ẩn dụ (0) | TN = 42,544 | FP = 1,388 |
| Ẩn dụ (1) | FN = 1,341 | TP = 4,902 |

Ma trận được suy ra từ số mẫu từng lớp và metric đầy đủ trong `commit3/history.json`, không phải từ file dự đoán từng dòng. Tính lại cả bốn metric khớp số gốc với sai số dưới 1e-12.

## False Positives (predicted metaphor, gold literal)

**1,388 FP**, chiếm **50.86% tổng lỗi**. Có **3.16% mẫu không ẩn dụ** bị dự đoán nhầm thành ẩn dụ; trong 6,290 dự đoán ẩn dụ, **22.07% là báo nhầm**. Hai tỷ lệ có mẫu số khác nhau, lần lượt là toàn bộ mẫu âm và toàn bộ dự đoán dương.

Trọng số lớp MD `[1, 4]` ưu tiên lỗi trên lớp ẩn dụ trong hàm mất mát. Đây là yếu tố nên kiểm tra khi phân tích đánh đổi precision–recall; chưa có ablation để kết luận nó gây ra các FP hiện tại.

## False Negatives (missed metaphors)

**1,341 FN**, chiếm **49.14% tổng lỗi**. Mô hình bỏ sót **21.48% số ẩn dụ thật**, tương đương khoảng một phần năm lớp dương.

Ở epoch 5, test recall đạt 83.26% nhưng precision chỉ 72.05%; đến epoch 8, precision tăng lên 77.93% và recall giảm còn 78.52%, trong khi F1 tăng từ 77.25% lên 78.23%. Kết quả cho thấy cần xét đồng thời precision và recall, thay vì chỉ ưu tiên phát hiện nhiều ẩn dụ hơn. Các điểm test này được mô tả hồi cứu, không dùng để chọn checkpoint.

## Nhận xét và các nhóm lỗi cần kiểm tra

- **Mất cân bằng nhãn:** 87.56% test là không ẩn dụ. Baseline luôn dự đoán 0 đã có Accuracy 87.56% nhưng F1 lớp ẩn dụ bằng 0. Vì vậy Accuracy 94.56% cần được đọc cùng F1 78.23% và tỷ lệ bỏ sót 21.48%.
- **Ngữ cảnh bị cắt:** `data_process.py` cắt chuỗi đầu vào ở 150 token. Cần kiểm tra các lỗi ở câu dài và vị trí từ mục tiêu trong phần ngữ cảnh còn lại; chưa đo được số lỗi liên quan đến việc cắt chuỗi.
- **Vị trí từ mục tiêu:** đầu vào dùng `lemma + sentence`; `target_loc` được đọc nhưng không được dùng để đánh dấu vị trí. Cần kiểm tra câu có nhiều lần xuất hiện cùng từ, đặc biệt khi các lần xuất hiện mang nghĩa khác nhau.
- **POS và nghĩa theo ngữ cảnh:** cần thống kê riêng FP/FN theo từ loại và đọc ví dụ trước khi kết luận model thường sai ở động từ, danh từ, giới từ hay các cách dùng đa nghĩa.

Hai nhóm lỗi FP/FN gần cân bằng về số lượng, nhưng tỷ lệ bỏ sót trên lớp ẩn dụ vẫn cao. Các nguyên nhân ngôn ngữ nêu trên là hướng kiểm chứng, chưa phải kết luận từ mẫu lỗi thực tế.

## Ví dụ FP/FN và kiểm tra file dự đoán

**Chưa có file dự đoán từng mẫu của AdMul trong output đã kiểm tra**, nên chưa thể trích 15 FP và 15 FN như báo cáo RoPPT/RNN_HG. Hàm `val()` hiện chỉ trả Accuracy, Precision, Recall và F1; không lưu `val_preds` ra file. Không dùng ví dụ của model khác làm lỗi của AdMul.

Để bổ sung phần này, cần xuất dự đoán từ `epoch_08.pt`, giữ đúng thứ tự test và lưu `row_id`, `target_loc`, `lemma`, `pos`, `sentence`, `gold`, `pred`. Trước khi lấy mẫu, kiểm tra:

- Đủ 50,175 dòng; `row_id` duy nhất và liên tục 0–50174.
- Mỗi dòng khớp câu, từ mục tiêu, vị trí và nhãn thật trong `VUA_All/test.tsv`.
- Metric và ma trận khớp TP=4902, FP=1388, FN=1341, TN=42544.
- Lấy 15 FP và 15 FN bằng seed cố định, lưu danh sách mẫu và giải thích từng lỗi.

## Tệp minh chứng

- [Kết quả cuối](commit3/result.json), [lịch sử 8 epoch](commit3/history.json), [cấu hình](commit3/config.yaml).
- [Số mẫu, phân bố nhãn và hash dữ liệu](commit1/data_check.json).
- [Kiểm tra tính nhất quán của kết quả](commit3/verification.json): đối chiếu log/JSON, chưa chạy inference độc lập từ checkpoint.

Khi so sánh với MelBERT/RoPPT trong repo, cần thống nhất tập token được đánh giá: AdMul hiện dùng 50,175 target test, trong khi hai báo cáo kia ghi 43,947 target. Chưa dùng các điểm này để xếp hạng chung giữa bốn model.
