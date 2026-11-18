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

Kiểm tra lại dữ liệu từ thư mục gốc repo bằng Python (chỉ dùng thư viện chuẩn):

```bash
python AdMul/scripts/check_data.py --data-root /path/to/data --output AdMul/results/commit1/data_check.json
```

`/path/to/data` là thư mục chứa `VUA_All/` và `semcor/`. Script kiểm tra dữ liệu và tài liệu trong mốc này do nhóm bổ sung; các file cấu hình được giữ nguyên từ tác giả.

Các tài liệu và kiểm tra dữ liệu được bổ sung khi đăng lại tiến trình lên GitHub; thời điểm commit không được coi là thời điểm chạy thí nghiệm trước đó.

## Mốc 2: notebook Kaggle và lưu/resume

- Mã mô hình, xử lý dữ liệu, sampler và train/eval ở thư mục này được sao chép nguyên bản từ commit tác giả đã ghi ở trên.
- Nhóm bổ sung `kaggle/runner.py`, `checkpointing.py`, `smoke.py`, `test_resume.py` và `build_notebook.py` để chạy nhiều phiên Kaggle, kiểm tra môi trường và khôi phục trạng thái huấn luyện.
- Notebook tự chứa mã nguồn: [AdMul_Kaggle_Checkpoints.ipynb](AdMul_Kaggle_Checkpoints.ipynb).
- Cách chạy phiên đầu và resume: [kaggle/README.md](kaggle/README.md).
- Minh chứng kiểm tra: [results/commit2/verification.json](results/commit2/verification.json).

Kết quả kiểm tra CPU ở mốc 2 dùng model nhỏ để kiểm tra cơ chế resume, không phải điểm thực nghiệm AdMul.

## Mốc 3: kết quả huấn luyện thực tế

Đã hoàn tất 8 epoch VUA All trên một Tesla T4. Epoch 8 có F1 validation tốt nhất (78,42%); F1 test tại epoch được chọn là **78,23%** (precision 77,93%, recall 78,52%). Chỉ dùng dev F1 để chọn checkpoint.

Log gốc, lịch sử từng epoch, cấu hình thực tế, nguồn output Kaggle và kiểm tra đối chiếu được lưu tại [results/commit3/](results/commit3/README.md). Model và dữ liệu lớn giữ ở Kaggle; không đưa vào GitHub.

## Các mốc tiếp theo

4. Đối chiếu kết quả với paper.
5. Phân tích lỗi AdMul.
6. So sánh các phương pháp của nhóm trên cùng split và metric.

Dữ liệu, model pretrained và checkpoint được lưu ngoài GitHub. Không đưa thông tin đăng nhập vào repo.
