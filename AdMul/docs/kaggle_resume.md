# Notebook Kaggle và cơ chế lưu/resume

## Phân biệt nguồn và đóng góp

Các file `model.py`, `data_process.py`, `data_utils.py`, `train_val.py`, `utils.py`, `main_vua.py`, `main_kf.py` và cấu hình được giữ nguyên từ commit `4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8` của SilasTHU/AdMul. Giấy phép được giữ tại `LICENSE`.

Nhóm bổ sung quy trình chạy Kaggle trong `kaggle/`:

| File | Vai trò |
|---|---|
| `build_notebook.py` | Đóng gói nguồn AdMul vào notebook; dùng danh sách file cụ thể để không đóng gói các dự án khác trong repo nhóm |
| `runner.py` | Nạp cấu hình VUA All, chuẩn bị dữ liệu, điều phối train/eval và giới hạn phiên |
| `checkpointing.py` | Lưu/khôi phục trạng thái; bỏ qua file tạm chưa lưu xong |
| `test_resume.py` | Kiểm tra tính nhất quán khi train liên tục và chia phiên bằng model nhỏ trên CPU |
| `smoke.py` | Kiểm tra GPU bằng kiến trúc AdMul với backbone thu nhỏ khởi tạo ngẫu nhiên; được gọi trong notebook |

Notebook đã được tạo sẵn ở thư mục `AdMul/`. Tạo lại từ thư mục gốc repo:

```bash
python AdMul/kaggle/build_notebook.py
```

## Trạng thái được lưu

Mỗi epoch hoàn chỉnh tạo `run/epochs/epoch_XX.pt` chứa model, optimizer AdamW, scheduler, RNG Python/NumPy/PyTorch/CUDA, số bước, lịch sử đánh giá và epoch tốt nhất theo dev F1. Dữ liệu tokenize chỉ lưu một lần trong `run/features.pt`.

Lưu qua file tạm, flush/fsync rồi đổi tên để không thay checkpoint hợp lệ bằng file ghi dở. Khi resume, so sánh identity của dữ liệu, nguồn code, model pretrained, cấu hình và môi trường; từ chối khi không khớp.

Nếu dừng giữa epoch, quay về epoch hoàn chỉnh gần nhất và chạy lại epoch dở. Không khôi phục vị trí batch giữa epoch. Phải giữ và gắn toàn bộ Output phiên trước khi chạy phiên sau.

## Thiết lập bổ sung so với mã gốc

- Cài Python 3.10.14, PyTorch 1.13.1+cu116; giữ Transformers 4.18.0 và NumPy 1.21.3 theo README tác giả. Môi trường cài tại `/tmp`.
- Khóa snapshot pretrained `microsoft/deberta-v3-base` tại revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
- Bật gradient checkpointing để giảm VRAM; giữ FP32, batch 64, không accumulation, dùng một GPU.
- Ghi các cờ deterministic/TF32 vào identity; giới hạn mặc định 3 epoch/phiên và 10,5 giờ tính từ đầu notebook.
- Giữ các lượt đánh giá dev/test/BSD-test mỗi epoch; chọn checkpoint chỉ bằng F1 dev.

## Kiểm tra cho commit này

Chạy từ thư mục gốc repo (môi trường kiểm tra cần PyTorch, NumPy, scikit-learn và tqdm):

```bash
python AdMul/kaggle/test_resume.py
```

Bốn bài kiểm tra CPU đã qua: train 3+3+2 epoch khớp train liền 8 epoch; compile cell và đối chiếu nguồn nhúng; từ chối đầu vào thay đổi và bỏ qua file tạm; xử lý dừng trước/giữa epoch.

Fixture dùng model nhỏ, thay tokenizer/optimizer và giả lập CUDA để chạy trên CPU. Nó kiểm tra cơ chế lưu/resume, không chứng minh điểm AdMul trên dữ liệu thật hay độ chính xác số học giữa các GPU.

Commit này không kèm log cài đặt, smoke test hay huấn luyện GPU Kaggle thực tế. Các log thực nghiệm và điểm số sẽ được bổ sung ở mốc tiếp theo.
