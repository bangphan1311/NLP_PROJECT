# Tái lập AdMul trên VUA All

Phần thực nghiệm cho bài toán nhận diện ẩn dụ trong NLP_PROJECT.

## Nguồn tham khảo

- Paper: [Adversarial Multi-task Learning for End-to-end Metaphor Detection](https://aclanthology.org/2023.findings-acl.96/), Findings of ACL 2023.
- Mã nguồn tác giả: https://github.com/SilasTHU/AdMul
- Commit nguồn: `4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8`.
- Giấy phép mã nguồn gốc: Apache-2.0, lưu tại `LICENSE`.

Mô hình và thuật toán AdMul thuộc công trình của tác giả. Công việc của nhóm là chuẩn bị thực nghiệm, tái lập trên Kaggle, ghi nhận kết quả và phân tích so sánh. Các phần bổ sung sẽ được ghi rõ ở từng mốc.

## Mốc 1: chuẩn bị tái lập

- Ghi nguồn paper và phiên bản mã nguồn.
- Lưu nguyên cấu hình mặc định và cấu hình VUA All của tác giả trong `configs/`.
- Kiểm tra số mẫu, phân bố nhãn và SHA-256 của các file dữ liệu cục bộ; kết quả trong `results/commit1/data_check.json`.
- Mô tả thiết lập tại [docs/experimental_setup.md](docs/experimental_setup.md).

Kiểm tra lại dữ liệu từ thư mục gốc repo bằng Python (chỉ dùng thư viện chuẩn):

```bash
python AdMul/scripts/check_data.py --data-root /path/to/data --output AdMul/results/commit1/data_check.json
```

`/path/to/data` là thư mục chứa `VUA_All/` và `semcor/`. Script kiểm tra dữ liệu và tài liệu trong mốc này do nhóm bổ sung; các file cấu hình được giữ nguyên từ tác giả.

Đây là commit chuẩn bị, chưa công bố kết quả huấn luyện. Các tài liệu và kiểm tra dữ liệu được bổ sung khi đăng lại tiến trình lên GitHub; thời điểm commit không được coi là thời điểm chạy thí nghiệm trước đó.

## Các mốc tiếp theo

1. Môi trường Kaggle và kiểm tra chạy thử.
2. Notebook cùng phần lưu checkpoint/resume được bổ sung.
3. Log, tiến độ và kết quả huấn luyện thực tế.
4. So sánh với paper/RNN_HG và phân tích lỗi.

Dữ liệu, model pretrained và checkpoint được lưu ngoài GitHub. Không đưa thông tin đăng nhập vào repo.
