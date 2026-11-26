# Tái lập AdMul và đối chiếu với bài báo

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

Mô hình có **184.429.834 tham số** (184,43 triệu), gồm backbone, pooler và các classifier/discriminator.
Nhóm bổ sung [scripts/count_parameters.py](scripts/count_parameters.py) để đếm từ metadata checkpoint,
chỉ dùng thư viện chuẩn Python, không cần PyTorch hay chạy lại huấn luyện.
Chạy từ thư mục gốc repo với checkpoint đã tải từ Kaggle:

```bash
python AdMul/scripts/count_parameters.py "/path/to/AdMulPaper/run/epochs/epoch_08.pt"
```

Nếu không truyền đường dẫn, script tìm checkpoint tại `AdMul/kaggle_output/result/AdMulPaper/run/epochs/epoch_08.pt`.
Script dành cho định dạng checkpoint AdMul/DeBERTa của thí nghiệm này; không tính tensor buffer `position_ids`
và không suy ra `requires_grad` từ `state_dict`.

## So sánh kết quả VUA All với paper

### Dữ liệu và cách đánh giá

Thực nghiệm dùng VUA All với **116.622 target train / 38.628 dev / 50.175 test**.
Đơn vị là target token được gán nhãn, không phải số câu. Chọn **epoch 8 bằng dev F1 cao nhất: 78,42%**.
Sau đó chạy inference từ checkpoint này để chia kết quả test theo cột POS; không huấn luyện lại.
Runner train có đánh giá test ở mỗi epoch, nhưng chỉ dùng dev F1 để chọn checkpoint.

Các chỉ số dưới đây dùng đơn vị **%**. Precision, Recall và F1 tính cho lớp ẩn dụ (`label=1`),
không phải macro-F1 giữa hai lớp. Chênh lệch được tính bằng **điểm phần trăm** từ các số đã làm tròn.

### Kết quả tổng thể

| VUA All | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| **Nhóm tái lập** | **77,93** | **78,52** | **78,23** | **94,56** |
| AdMul trong paper | 78,40 | 79,50 | 79,00 | 94,70 |
| Nhóm − paper | −0,47 | −0,98 | −0,77 | −0,14 |

Nguồn đối chiếu: [Bảng 4 của paper, trang 1489](https://aclanthology.org/2023.findings-acl.96.pdf#page=7).
F1 tái lập thấp hơn khoảng **0,77 điểm**, với mức giảm recall lớn hơn precision.

### Kết quả theo từ loại

| POS | Số target test | Kết quả | Precision | Recall | F1 | Accuracy |
|---|---:|---|---:|---:|---:|---:|
| VERB | 9.872 | **Nhóm** | **75,62** | **76,95** | **76,28** | **90,85** |
| VERB | — | Paper | 77,20 | 78,10 | 77,60 | 91,40 |
| ADJ | 3.965 | **Nhóm** | **71,69** | **65,06** | **68,22** | **91,73** |
| ADJ | — | Paper | 72,40 | 66,90 | 69,50 | 92,00 |
| ADV | 3.393 | **Nhóm** | **75,22** | **70,90** | **73,00** | **96,23** |
| ADV | — | Paper | 76,30 | 71,30 | 73,70 | 96,30 |
| NOUN | 8.588 | **Nhóm** | **76,52** | **68,16** | **72,10** | **92,07** |
| NOUN | — | Paper | 77,00 | 70,30 | 73,50 | 92,40 |

Nguồn số công bố: [Bảng 5 của paper](https://aclanthology.org/2023.findings-acl.96.pdf#page=7).
Số target của nhóm được đối chiếu trực tiếp với cùng file test TSV của lần train.

| Nhóm − paper | Δ Precision | Δ Recall | Δ F1 | Δ Accuracy |
|---|---:|---:|---:|---:|
| VERB | −1,58 | −1,15 | **−1,32** | −0,55 |
| ADJ | −0,71 | −1,84 | **−1,28** | −0,27 |
| ADV | −1,08 | −0,40 | **−0,70** | −0,07 |
| NOUN | −0,48 | −2,14 | **−1,40** | −0,33 |

NOUN có chênh lệch F1 lớn nhất (**−1,40 điểm**), đồng thời recall giảm nhiều nhất (**−2,14 điểm**).
Trong kết quả tái lập, VERB có F1 cao nhất (76,28%), còn ADJ thấp nhất (68,22%).
ADV có Accuracy 96,23% nhưng F1 73,00%; không dùng riêng Accuracy để kết luận nhóm nào dễ hơn
vì phân bố nhãn giữa các nhóm có thể khác nhau.

Bốn POS trên có tổng **25.818 target**. ALL còn chứa các POS khác nên F1 ALL không phải trung bình
của bốn F1 này. Nhóm VERB 9.872 target trong VUA All **khác benchmark VUA Verb test 5.873 target**
với split và lượt train riêng.

### Tham khảo các phương pháp trong paper

| Phương pháp | VERB F1 | ADJ F1 | ADV F1 | NOUN F1 |
|---|---:|---:|---:|---:|
| RNN_ELMo | 69,9 | 58,3 | 59,7 | 60,4 |
| RNN_BERT | 69,5 | 54,7 | 62,9 | 59,9 |
| RNN_HG | 70,7 | 62,2 | 63,8 | 63,4 |
| RNN_MHCA | 70,7 | 61,6 | 63,2 | 63,2 |
| DeepMet | 73,3 | 63,3 | 72,3 | 65,4 |
| MelBERT | 75,1 | 64,4 | 74,6 | 70,7 |
| MisNet | 77,6 | 67,0 | 73,3 | 70,6 |
| AdMul — paper | 77,6 | 69,5 | 73,7 | 73,5 |
| **AdMul — nhóm tái lập** | **76,28** | **68,22** | **73,00** | **72,10** |

Nguồn các hàng tham khảo: [Bảng 5 của paper](https://aclanthology.org/2023.findings-acl.96.pdf#page=7).
Chỉ hàng cuối là kết quả AdMul nhóm vừa chạy; các hàng còn lại **không phải kết quả nhóm chạy lại**.
Các baseline RNN là mạng nơ-ron, không thay thế thực nghiệm phương pháp truyền thống trên cùng split.

### Thời gian, số tham số và phần cứng

| Hạng mục | Ghi nhận của lần tái lập VUA All |
|---|---|
| Số epoch / seed | 8 / 4 |
| Thời gian | **36,62 giờ**, tổng thời gian epoch gồm train và các lượt đánh giá |
| Số tham số | **184.429.834**, không gồm optimizer, scheduler và buffer `position_ids` |
| GPU | **1 Tesla T4**; chỉ dùng GPU 0 |
| CUDA / Python | 11.6 / 3.10.14 |
| PyTorch / Transformers / NumPy | 1.13.1+cu116 / 4.18.0 / 1.21.3 |
| Cấu hình chính | Batch 64, LR 3e-5, warmup 2 epoch, max length 150, dropout 0 |
| Bộ nhớ/tính toán | Gradient checkpointing bật, FP32, không AMP |

Thời gian lấy từ tổng trường `seconds` trong [history.json](results/commit3/history.json),
không bao gồm đầy đủ tải/cài môi trường, chờ giữa phiên, lưu checkpoint và các phiên bị gián đoạn.
Chưa tách thời gian train thuần hay ghi peak VRAM cho lần train này. Paper dùng RTX 3090;
chưa có đối chứng để quy chênh lệch điểm cho phần cứng hoặc so sánh tốc độ trực tiếp.

### Minh chứng và giới hạn

- [Kết quả POS thực tế](results/vua_all_pos/metrics_by_pos.csv) và [bảng chênh lệch](results/vua_all_pos/paper_comparison.csv).
- [Log inference](results/vua_all_pos/metrics_by_pos.csv.log), [log phiên Kaggle](results/vua_all_pos/vua-all-pos.log) và [mã evaluator đã chạy](results/vua_all_pos/evaluate_vua_all_used.py).
- [Kiểm tra số mẫu, điểm ALL và SHA-256](results/vua_all_pos/verification.json); [thông tin môi trường train](results/commit3/identity.json).
- Nguồn Output: [Kaggle VUA All POS](https://www.kaggle.com/code/lminhhinlminhhin/vua-all-pos).

Điểm ALL của lần tính POS khớp điểm epoch 8 đã lưu. Đây mới là một seed nên chưa có khoảng tin cậy
hoặc kiểm định ý nghĩa thống kê. Notebook POS chưa xuất dự đoán từng mẫu; chưa thể đưa ví dụ FP/FN
theo POS hay khẳng định nguyên nhân ngôn ngữ của các lỗi từ bảng điểm tổng hợp.

Các báo cáo MelBERT/RoPPT của nhóm hiện ghi **43.947 target test**, còn AdMul/RNN_HG ghi **50.175**.
Cần đối chiếu câu, vị trí từ, nhãn và cùng tập target trước khi xếp hạng kết quả thực nghiệm chung.
Phần đối chiếu với paper ở trên chưa đáp ứng đầy đủ yêu cầu chạy mọi phương pháp của nhóm
và baseline truyền thống trên cùng dữ liệu/split.

## So sánh kết quả TroFi với paper

Thực nghiệm từ [Output Kaggle lminhhinlminhhin/trofi](https://www.kaggle.com/code/lminhhinlminhhin/trofi)
đã hoàn tất **10/10 fold, 10 epoch mỗi fold, seed 4**, trên **3.737 target** (1.627 nhãn ẩn dụ, 2.110 nhãn nghĩa đen).
Sau một lần shuffle, runner chia 10 đoạn liên tiếp theo `main_kf.py`: chín fold có 373 target held-out,
fold cuối có 380; không chia stratified. P/R/F1 tính cho lớp ẩn dụ (`label=1`).

| Kết quả TroFi | Precision (%) | Recall (%) | F1 (%) | Accuracy (%) |
|---|---:|---:|---:|---:|
| AdMul — paper, Bảng 4 | 70,50 | 79,80 | 74,70 | 76,50 |
| **Nhóm — epoch tốt nhất theo held-out, trung bình 10 fold** | **69,23** | **77,97** | **73,08** | **75,01** |
| Nhóm − paper, điểm phần trăm | −1,27 | −1,83 | **−1,62** | −1,49 |
| **Nhóm — epoch 10 cố định, trung bình 10 fold** | **70,84** | **68,86** | **69,58** | **73,83** |
| Nhóm − paper, điểm phần trăm | +0,34 | −10,94 | **−5,12** | −2,67 |

Nguồn số công bố: [Bảng 4, trang 1489 của paper](https://aclanthology.org/2023.findings-acl.96.pdf#page=7).
Chênh lệch tính từ số đã làm tròn hai chữ số. F1 tái lập đạt **73,08%**, thấp hơn paper **1,62 điểm**.
Precision và recall đều thấp hơn số công bố; với epoch cuối, recall giảm rõ hơn, còn precision gần mức paper.

**Quy tắc chọn epoch cần được giữ rõ:** code tác giả chọn epoch có F1 cao nhất trên chính held-out fold,
rồi lấy trung bình metric của 10 fold. Mức 73,08% dùng quy tắc này, không có tập dev độc lập để chọn checkpoint.
Mức 69,58% dùng epoch 10 cố định, tránh chọn epoch bằng nhãn held-out. Chênh lệch giữa hai cách là **3,50 điểm F1**;
chưa thể dùng chênh lệch này để kết luận nguyên nhân hay ý nghĩa thống kê của khoảng cách với paper.
Đây là huấn luyện TroFi theo cross-validation, đối chiếu Bảng 4; không phải zero-shot ở Bảng 8.

Tổng thời gian train **13,34 giờ**, train + đánh giá **13,93 giờ** trên **1 Tesla T4**;
mỗi model có **184.429.834 tham số**, peak GPU memory allocated lớn nhất **5,64 GiB**.
Cấu hình chính: DeBERTa-v3-base, batch 64, LR `1e-5`, max length 150, gradient checkpointing bật, FP32.
Epoch/batch/LR khớp [Bảng 3 của paper](https://aclanthology.org/2023.findings-acl.96.pdf#page=6).
Paper dùng RTX 3090; chưa có thí nghiệm đối chứng để quy khoảng cách điểm cho phần cứng.

Đã kiểm tra đủ 3.737 row ID xuất hiện đúng một lần trong dự đoán ngoài fold, phân hoạch train/held-out không giao nhau,
metric từng fold và trung bình khớp JSON, epoch tốt nhất khớp lịch sử, SHA-256 dữ liệu khớp metadata lần chạy.
F1 gộp dự đoán epoch cuối là **69,59%**, khác trung bình F1 theo fold **69,58%** do cách tổng hợp.

- [Báo cáo TroFi đầy đủ: từng fold, cấu hình, giới hạn](results/trofi/README.md).
- [Kết quả gốc](results/trofi/result.json), [bảng so sánh CSV](results/trofi/paper_comparison.csv), [kiểm tra số liệu](results/trofi/verification.json).
- [Nguồn Output và SHA-256](results/trofi/source.json), [dự đoán ngoài fold của epoch cuối](results/trofi/predictions_oof_final.csv).

## Các công việc còn lại

- Xuất dự đoán từng mẫu và phân tích lỗi AdMul.
- Thống nhất dữ liệu, split và metric để so sánh các phương pháp của nhóm.
- Bổ sung phương pháp truyền thống theo yêu cầu bài tập.

Dữ liệu, model pretrained và checkpoint được lưu ngoài GitHub. Không đưa thông tin đăng nhập vào repo.
