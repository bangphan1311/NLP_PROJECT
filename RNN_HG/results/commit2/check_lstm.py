
import sys
import json
import torch

torch.backends.cudnn.enabled = True

info = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cudnn_enabled": torch.backends.cudnn.enabled,
    "cuda_available": torch.cuda.is_available()
}
print(json.dumps(info, indent=2), flush=True)

assert torch.__version__ == sys.argv[1]
assert torch.version.cuda.startswith(sys.argv[2])
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1

gpu = torch.cuda.get_device_name(0)
print("GPU:", gpu, flush=True)
assert "T4" in gpu
print("ENV_READY", flush=True)

model = torch.nn.LSTM(
    input_size=1324,
    hidden_size=256,
    batch_first=True,
    bidirectional=True
).cuda()

x = torch.randn(2, 18, 1324).cuda()
y, _ = model(x)
loss = y.sum()
loss.backward()
torch.cuda.synchronize()

print("LSTM_GPU_OK", flush=True)
