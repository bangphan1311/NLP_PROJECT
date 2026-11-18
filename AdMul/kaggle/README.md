# Chạy AdMul trên Kaggle, lưu đủ trạng thái mỗi epoch

Upload `AdMul/AdMul_Kaggle_Checkpoints.ipynb` từ repo NLP_PROJECT vào Kaggle.
Notebook tự chứa mã nguồn, không cần upload các file Python riêng.

## Phiên đầu

1. Bật GPU T4 và Internet. Notebook dùng một GPU.
2. Tạo Dataset từ `admul_data.zip` gốc trong Downloads rồi Add Input Dataset đó.
   Cũng nhận thư mục dữ liệu đã giải nén hoặc Output `lminhhinlminhhin/full-data` để lấy dữ liệu.
3. Giữ `MODE = 'auto'`, `SEED = 4`, `MAX_SESSION_EPOCHS = 3`, `SESSION_HOURS = 10.5`.
4. Chọn **Save Version → Save & Run All**.

## Phiên sau

1. Gỡ Input cũ; Add Input → Notebook → Output phiên gần nhất của notebook này.
2. Giữ nguyên code và cấu hình. Có thể đặt `MODE = 'resume'` để bắt buộc có checkpoint.
3. Save & Run All. Dự kiến các phiên chạy epoch 1–3, 4–6, 7–8.
4. Log phải có `event: resume`, `optimizer_restored: true`, số epoch đã hoàn thành đúng.

Không gắn nhiều bản Output. Không dùng checkpoint cũ chỉ có trọng số để nối lần train này.
Nếu MODE auto không thấy Output của notebook này, nó sẽ coi đây là phiên đầu và chạy từ đầu.

## File cần giữ

Giữ toàn bộ thư mục Output `AdMulPaper`:

| Đường dẫn | Nội dung |
|---|---|
| `run/epochs/epoch_XX.pt` | Mỗi epoch một file đầy đủ: model, optimizer, scheduler, RNG, tiến độ, lịch sử |
| `run/features.pt` | Các tập tokenize và thứ tự BSD đã cắt/shuffle, lưu một lần |
| `data/data` | Dữ liệu gốc, giữ một bản để kiểm tra và tái sử dụng |
| `deberta-v3-base` | Pretrained snapshot cố định |
| `src` | Mã tác giả và runner được dùng trong thí nghiệm |
| `run/identity.json` | Hash các đầu vào và phiên bản môi trường |
| `run/result.json` | Tiến độ, epoch tốt nhất, đường dẫn checkpoint, điểm test của epoch đó |
| `run/history.json`, `events.jsonl`, `console.log` | Điểm từng epoch và log |

Checkpoint được ghi ra file tạm, flush/fsync, rồi đổi tên. Giữ các epoch cũ khi tạo epoch mới.
File `.tmp` không được coi là checkpoint hoàn chỉnh. Không tạo ZIP lặp lại các checkpoint lớn.
Các phiên sau sao chép toàn bộ Output trước sang `/kaggle/working` rồi nối tiếp.
Nếu ngắt giữa epoch, checkpoint cuối epoch trước vẫn là điểm khởi động lại;
không khôi phục tiến độ bên trong epoch. Mốc 10,5 giờ và giới hạn ba epoch giúp
notebook kết thúc bình thường trước timeout; không bảo đảm lấy lại Output nếu Kaggle hủy đột ngột.

## Giữ cách train sát repo/paper

Nguồn tác giả: commit `4847c72a4f8228a9197d1cbd0664eab7ddb0cfc8` của
[SilasTHU/AdMul](https://github.com/SilasTHU/AdMul).
[Paper](https://aclanthology.org/2023.findings-acl.96/).

Không sửa model, processor, sampler, train/val, padding/truncation và loss gốc.
VUA All dùng toàn bộ split; BSD cắt SemCor bằng số dòng VUA train trước khi shuffle như tác giả.
Giữ cả ba lượt đánh giá dev/test/BSD-test mỗi epoch, vì bỏ một lượt DataLoader có thể đổi RNG.
Chỉ chọn model bằng F1 dev.

Transformers 4.18.0 và NumPy 1.21.3 đúng README gốc; dùng Transformers AdamW gốc,
không thay bằng PyTorch AdamW. Python 3.10.14 và torch 1.13.1+cu116 là môi trường được chọn
vì repo không chỉ rõ hai phiên bản này. CUDA 11.6 theo paper, nhưng Kaggle T4 khác RTX 3090
được dùng trong paper. Cài môi trường vào `/tmp` để không ghi dependency vào Output.
Pretrained `microsoft/deberta-v3-base` khóa revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.

Khác code gốc: gradient checkpointing, giữ RNG khi tính lại, để giảm VRAM;
các cờ deterministic/TF32 được ghi trong identity. Vẫn FP32, batch 64, không AMP,
không accumulation, không chia batch qua hai GPU và không tự thay tham số khi OOM.
Không bảo đảm điểm giống hệt paper do thiếu môi trường gốc đầy đủ và sai khác số học GPU.
Resume từ chối khi dữ liệu, code, features, cấu hình, runtime hoặc loại GPU thay đổi.

## Kiểm tra cho commit 2

`python AdMul/kaggle/test_resume.py` (từ thư mục gốc repo): bốn bài kiểm tra CPU, dùng sampler, train và val thật của tác giả
với model nhỏ; thay tokenizer/optimizer cho fixture kiểm tra. Kiểm tra chạy 3+3+2 epoch
khớp chạy liền 8 epoch về model, optimizer, scheduler, RNG và metric; giữ đủ tám checkpoint;
từ chối dữ liệu thay đổi; bỏ qua file ghi dở; dừng trước epoch và giữa epoch không làm mất checkpoint trước.
Đã compile mọi cell notebook và kiểm tra snapshot code khớp file nguồn.

Commit này chưa kèm minh chứng chạy GPU Kaggle thực tế.
Notebook sẽ báo lỗi nếu cài đặt/CUDA không hoạt động, không tự đổi sang phiên bản khác.
Notebook có dependency `six==1.16.0` và import tường minh `torch.utils.checkpoint` cho Transformers 4.18.
Trước khi chuẩn bị dữ liệu lớn, notebook chạy `smoke.py` với model AdMul thật,
backbone DeBERTa thu nhỏ ngẫu nhiên, một bước train có gradient checkpointing,
backward/AdamW và một lượt evaluation trên GPU. Cần thấy `SMOKE OK` để xác nhận bước này qua.
Smoke test này sẽ chạy trên Kaggle, chưa được thực thi trên GPU trong workspace local.

Tạo lại notebook sau khi sửa runner: `python AdMul/kaggle/build_notebook.py` (từ thư mục gốc repo).
