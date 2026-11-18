"""Build the uploadable notebook; keep author files unchanged."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Explicit manifest: building from the group repo must not embed other projects
# or label the group's HEAD as the original author's revision.
commit = json.loads((ROOT / 'results/commit1/source.json').read_text(encoding='utf-8'))['commit']
author_files = [
    'LICENSE', 'model.py', 'data_process.py', 'data_utils.py', 'train_val.py',
    'utils.py', 'main_vua.py', 'main_kf.py', 'configs/default.py',
    'configs/vua_all.yaml', 'configs/vua_verb.yaml', 'configs/mohx.yaml',
    'configs/trofi.yaml',
]
sources = {name: (ROOT / name).read_text(encoding='utf-8') for name in author_files}
sources.update({name: (ROOT / 'kaggle' / name).read_text(encoding='utf-8')
                for name in ['runner.py', 'checkpointing.py', 'smoke.py']})


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))


def code(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=text.splitlines(keepends=True))


intro = """# AdMul VUA All — checkpoint đầy đủ mỗi epoch

## Lần đầu
1. Import notebook này. Bật **GPU T4** và **Internet**. Chỉ dùng **một GPU**, FP32.
2. **Add Input** Dataset chứa `admul_data.zip` (bản ZIP dữ liệu gốc), hoặc dữ liệu đã giải nén.
   Có thể gắn Output `lminhhinlminhhin/full-data` để lấy dữ liệu; checkpoint cũ sẽ không được nạp.
3. Giữ `MODE = 'auto'`, seed 4, tối đa 3 epoch/phiên. **Save Version → Save & Run All**.

## Các phiên sau
1. Gỡ Input cũ, gắn **Output phiên gần nhất của notebook này** (chỉ một bản).
2. Giữ nguyên notebook và `MODE = 'auto'`, rồi **Save & Run All**.
3. Tự khôi phục từ epoch đã hoàn thành: dự kiến **1–3 → 4–6 → 7–8**.
   Nếu thiếu checkpoint khi bạn đã chạy trước đó, đặt `MODE = 'resume'` để bắt buộc báo lỗi,
   tránh vô tình bắt đầu lại. Không tự khôi phục được nếu bạn không gắn Output phiên trước.

## Được giữ lại trong Output
- `AdMulPaper/run/epochs/epoch_01.pt`, `epoch_02.pt`, …: **tất cả checkpoint epoch**,
  gồm model, AdamW, scheduler, RNG Python/NumPy/PyTorch/CUDA, bước train, lịch sử và epoch tốt nhất.
- `data/data/`: dữ liệu gốc; `run/features.pt`: dữ liệu tokenize, gồm thứ tự BSD đã shuffle,
  lưu một lần, không lặp lại dữ liệu lớn trong mỗi checkpoint.
- `run/identity.json`: SHA-256 dữ liệu, code, features và pretrained model; phiên bản môi trường.
  Resume từ chối khi đầu vào, cấu hình, môi trường hoặc GPU khác.
- `run/history.json`, `result.json`, `events.jsonl`, `console.log`, tokenizer, pretrained model và code.
  Model tốt nhất được chỉ ra bởi `best_checkpoint` trong `result.json`.

## Sát paper và code gốc
Giữ nguyên model, processor, sampler, train/val và cấu hình VUA All từ repo tác giả.
Transformers **4.18.0**, NumPy **1.21.3**, **Transformers AdamW gốc**, 8 epoch, batch 64,
LR 3e-5, warmup 2 epoch, max length 150, seed 4, dropout 0, loss và lịch adversarial gốc.
Giữ toàn bộ VUA train/dev/test. BSD lấy số dòng bằng VUA train và shuffle sau khi cắt,
đúng `main_vua.py`; không tự dùng thêm SemCor dư hay thay cách tokenize/cắt câu.
Giữ ba lượt đánh giá mỗi epoch như tác giả; chỉ dùng **F1 dev** để chọn model.

**Khác biệt được ghi rõ:** bật gradient checkpointing để giảm activation, giữ RNG khi tính lại;
không dùng AMP, gradient accumulation hay multi-GPU. Dùng Python 3.10.14/PyTorch 1.13.1 CUDA 11.6
vì repo không công bố phiên bản Python/PyTorch. Pretrained DeBERTa-v3-base được khóa revision;
paper ghi GPU RTX 3090/CUDA 11.6; Kaggle T4 khác GPU và paper không cung cấp đầy đủ
trạng thái môi trường để bảo đảm trùng điểm tuyệt đối.
Các cờ deterministic được bật (warn-only cho phép toán chưa hỗ trợ); GPU vẫn có thể có sai khác số học.
Không bảo đảm điểm số y hệt paper; mục tiêu là giữ thuật toán và khôi phục trạng thái đầy đủ.

Notebook cài môi trường Python riêng tại `/tmp` để chạy thư viện cũ trên Kaggle mới.
Nếu cài đặt hoặc CUDA không tương thích, sẽ báo lỗi; không tự đổi sang thư viện mới.
Giới hạn phiên 10,5 giờ tính từ cell đầu, tối đa 3 epoch; kiểm tra thời gian trước mỗi epoch
và giữa các batch. Khi hết ngân sách, thoát bình thường để giữ Output; epoch dở sẽ chạy lại
từ checkpoint epoch hoàn chỉnh trước đó. Nếu Kaggle cưỡng bức dừng trước khi lưu Output,
không thể bảo đảm lấy lại file, nên dùng Save & Run All và giữ khoảng đệm thời gian.

Mỗi full checkpoint khoảng vài GB; giữ 8 epoch cần nhiều dung lượng. Không tạo ZIP lặp checkpoint.
Chỉ nạp các file `.pt` do chính bạn tạo. File chứa cả dữ liệu pickle của optimizer và RNG.

Nguồn: [paper](https://aclanthology.org/2023.findings-acl.96/),
[repo tác giả](https://github.com/SilasTHU/AdMul),
[Transformers 4.18 optimizer](https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/optimization.py).
"""
settings = """# Đọc lại cell này ở mỗi phiên. Không cần đổi MODE nếu đã gắn đúng Output gần nhất.
import os, sys, time, json, shutil, subprocess, zipfile
from pathlib import Path
SESSION_STARTED = time.monotonic()
MODE = 'auto'   # auto: nối khi có checkpoint của notebook này, lần đầu chạy từ epoch 1.
                # resume: bắt buộc có checkpoint; fresh: bắt đầu thí nghiệm mới.
SEED = 4
MAX_SESSION_EPOCHS = 3
SESSION_HOURS = 10.5
INPUT_ROOT = Path('/kaggle/input')
WORK = Path('/kaggle/working/AdMulPaper')
assert MODE in ('auto', 'resume', 'fresh')
assert not WORK.exists(), 'Working directory đã tồn tại. Khởi động phiên Kaggle mới để tránh trộn kết quả.'
WORK.mkdir(parents=True)
markers = list(INPUT_ROOT.rglob('admul_run.json'))
if MODE == 'fresh':
    markers = []
if len(markers) > 1:
    raise ValueError(f'Có nhiều bản Output: {markers}. Chỉ gắn phiên gần nhất.')
if MODE == 'resume' and not markers:
    raise FileNotFoundError('Gắn Output phiên gần nhất có admul_run.json trước khi resume')
if markers:
    previous = markers[0].parent
    print('Restore workspace:', previous)
    shutil.copytree(previous, WORK, dirs_exist_ok=True)
    if MODE == 'resume' and not list((WORK / 'run/epochs').glob('epoch_*.pt')):
        raise FileNotFoundError('Output này chưa có epoch hoàn chỉnh để resume')
else:
    print('FRESH RUN: pretrained backbone only; no previous trained checkpoint')
"""
data_setup = """# Khôi phục từ ZIP gốc hoặc thư mục dữ liệu, bỏ metadata macOS.
if not (WORK / 'data/data/VUA_All/train.tsv').exists():
    archives = list(INPUT_ROOT.rglob('admul_data.zip'))
    if len(archives) > 1:
        raise ValueError('Chỉ gắn một bản admul_data.zip')
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            for item in archive.infolist():
                parts = Path(item.filename).parts
                if (item.is_dir() or not parts or parts[0] != 'data'
                        or item.filename.startswith('__MACOSX') or parts[-1] == '.DS_Store'):
                    continue
                destination = (WORK / 'data').joinpath(*parts).resolve()
                if not destination.is_relative_to(WORK.resolve()):
                    raise ValueError('Unsafe archive path')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, destination.open('wb') as target:
                    shutil.copyfileobj(source, target)
    else:
        candidates = [p.parent.parent for p in INPUT_ROOT.rglob('train.tsv')
                      if p.parent.name == 'VUA_All'
                      and (p.parent.parent / 'semcor/threshold/train_3.tsv').exists()]
        if len(candidates) != 1:
            raise FileNotFoundError(f'Gắn đúng một Dataset dữ liệu gốc; tìm thấy {candidates}')
        shutil.copytree(candidates[0], WORK / 'data/data')
assert (WORK / 'data/data/VUA_All/train.tsv').exists(), 'Thiếu VUA All'
assert (WORK / 'data/data/semcor/threshold/train_3.tsv').exists(), 'Thiếu SemCor'
"""
source_setup = 'SOURCES = ' + repr(sources) + '\nSOURCE_COMMIT = ' + repr(commit) + '\n' + """for name, content in SOURCES.items():
    destination = WORK / 'src' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text(encoding='utf-8') != content:
        raise ValueError(f'Code thay đổi giữa các phiên: {name}. Dùng cùng notebook để resume.')
    destination.write_text(content, encoding='utf-8')
marker = WORK / 'admul_run.json'
if not marker.exists():
    marker.write_text(json.dumps(dict(format='admul-epoch-v1', source_commit=SOURCE_COMMIT)), encoding='utf-8')
print('Author source commit:', SOURCE_COMMIT)
"""
environment = """# Môi trường cũ nằm ngoài /kaggle/working, không chiếm dung lượng Output.
subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv==0.6.17'], check=True)
uv = [sys.executable, '-m', 'uv']
env_root = Path('/tmp/admul-paper-env')
subprocess.run(uv + ['venv', '--python', '3.10.14', '--seed', str(env_root)], check=True)
PYTHON = str(env_root / 'bin/python')
subprocess.run(uv + ['pip', 'install', '--python', PYTHON, '--no-cache',
    'torch==1.13.1+cu116', '--extra-index-url', 'https://download.pytorch.org/whl/cu116',
    '--index-strategy', 'unsafe-best-match',
    'transformers==4.18.0', 'numpy==1.21.3', 'tokenizers==0.12.1',
    'huggingface-hub==0.10.1', 'sentencepiece==0.1.99', 'protobuf==3.20.3', 'six==1.16.0',
    'scikit-learn==1.2.2', 'scipy==1.10.1', 'yacs==0.1.8', 'tqdm==4.66.5',
    'filelock==3.13.1', 'packaging==23.2', 'pyyaml==6.0.1', 'regex==2023.12.25',
    'requests==2.31.0', 'sacremoses==0.1.1', 'typing-extensions==4.9.0',
    'joblib==1.3.2', 'threadpoolctl==3.2.0', 'click==8.1.7',
    'certifi==2024.2.2', 'charset-normalizer==3.3.2', 'idna==3.6', 'urllib3==2.2.1'], check=True)
child_env = os.environ.copy()
child_env.update(CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 PYTHONHASHSEED=str(SEED), TOKENIZERS_PARALLELISM='false',
                 USE_TF='0', USE_FLAX='0', HF_HOME='/tmp/admul-hf')
subprocess.run([PYTHON, '-c',
    'import torch,transformers,numpy; print(torch.__version__,transformers.__version__,numpy.__version__); '
    'import six; print("six ready:", six.__version__); '
    'import torch.utils.checkpoint; '
    'from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer; '
    'assert torch.cuda.is_available(), "CUDA không hoạt động trong môi trường riêng"; '
    'print(torch.cuda.get_device_name(0))'], env=child_env, check=True)
(WORK / 'run').mkdir(exist_ok=True)
lock_path = WORK / 'run/requirements.lock.txt'
new_lock = subprocess.check_output([PYTHON, '-m', 'pip', 'freeze'], text=True)
if lock_path.exists() and lock_path.read_text() != new_lock:
    raise RuntimeError('Dependency lock changed. Do not resume with different dependencies.')
lock_path.write_text(new_lock)
subprocess.run([PYTHON, '-u', str(WORK / 'src/smoke.py')], cwd=WORK, env=child_env, check=True)
"""
model_setup = """# Snapshot bất biến; không dùng model đã fine-tune từ Input cũ.
import urllib.request
MODEL_ID = 'microsoft/deberta-v3-base'
REVISION = '8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
backbone = WORK / 'deberta-v3-base'
backbone.mkdir(exist_ok=True)
for filename in ['config.json', 'pytorch_model.bin', 'spm.model', 'tokenizer_config.json']:
    destination = backbone / filename
    if not destination.exists():
        print('Downloading pretrained:', filename, flush=True)
        request = urllib.request.Request(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{filename}',
                                         headers={'User-Agent': 'AdMul-reproduction'})
        temporary = destination.with_suffix(destination.suffix + '.part')
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output, length=8*1024*1024)
        os.replace(temporary, destination)
(backbone / 'source_revision.txt').write_text(f'{MODEL_ID}\\n{REVISION}\\n', encoding='utf-8')
"""
execute = """remaining = SESSION_HOURS * 3600 - (time.monotonic() - SESSION_STARTED)
if remaining <= 0:
    raise RuntimeError('Hết ngân sách phiên trong bước chuẩn bị. Chạy lại trong phiên mới.')
command = [PYTHON, '-u', str(WORK / 'src/runner.py'), '--work', str(WORK),
           '--seed', str(SEED), '--max-session-epochs', str(MAX_SESSION_EPOCHS),
           '--seconds-left', str(remaining)]
with (WORK / 'run/console.log').open('a', encoding='utf-8') as log:
    process = subprocess.Popen(command, cwd=WORK, env=child_env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
if returncode:
    raise RuntimeError(f'Runner lỗi {returncode}; xem console.log. Không tự giảm batch hay đổi cấu hình.')
"""
finish = """result = json.loads((WORK / 'run/result.json').read_text())
print(json.dumps(result, indent=2))
print('Checkpoint epochs:', [p.name for p in sorted((WORK / 'run/epochs').glob('epoch_*.pt'))])
if result['training_complete']:
    print('Đã xong 8 epoch. Model tốt nhất:', result['best_checkpoint'])
else:
    print('Phiên đã dừng. Gắn Output phiên này vào phiên mới để chạy epoch', result['next_epoch'])
print('Giữ toàn bộ thư mục AdMulPaper trong Output, không chỉ một file .pt.')
"""
notebook = dict(nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python'),
                              kaggle=dict(isInternetEnabled=True, accelerator='gpu', isGpuEnabled=True)),
                cells=[md(intro), code(settings), code(data_setup), code(source_setup),
                       code(environment), code(model_setup), code(execute), code(finish)])
for i, cell in enumerate(notebook['cells']):
    cell['id'] = 'admul-%02d' % i
path = ROOT / 'AdMul_Kaggle_Checkpoints.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(path)
