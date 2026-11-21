"""Count AdMul parameters from checkpoint metadata, without PyTorch or training."""
import argparse
from collections import OrderedDict
import codecs
import io
import math
from pathlib import Path
import pickle
import zipfile


class TensorMetadata:
    def __init__(self, storage, offset, shape, stride, *unused):
        self.shape = shape

    def numel(self):
        return math.prod(self.shape)


class IgnoredMetadata:
    """Ignore storage and NumPy RNG metadata; never load their array contents."""

    def __init__(self, *args):
        pass

    def __setstate__(self, state):
        pass


class MetadataReader(pickle.Unpickler):
    # Only these known checkpoint constructors are permitted. No model code runs.
    ALLOWED = {
        ("collections", "OrderedDict"): OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): TensorMetadata,
        ("torch", "FloatStorage"): IgnoredMetadata,
        ("torch", "LongStorage"): IgnoredMetadata,
        ("torch", "ByteStorage"): IgnoredMetadata,
        ("torch.torch_version", "TorchVersion"): str,
        ("numpy", "dtype"): IgnoredMetadata,
        ("numpy", "ndarray"): IgnoredMetadata,
        ("numpy.core.multiarray", "_reconstruct"): IgnoredMetadata,
        ("_codecs", "encode"): codecs.encode,
    }

    def find_class(self, module, name):
        try:
            return self.ALLOWED[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f"Unsupported metadata: {module}.{name}") from None

    def persistent_load(self, storage_id):
        return None  # Tensor values are unnecessary for counting shapes.


def main():
    default = (Path(__file__).resolve().parents[1] / "kaggle_output" / "result"
               / "AdMulPaper" / "run" / "epochs" / "epoch_08.pt")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=default)
    args = parser.parse_args()

    with zipfile.ZipFile(args.checkpoint) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        checkpoint = MetadataReader(io.BytesIO(archive.read(metadata))).load()

    state = checkpoint["model"]
    # For this AdMul/DeBERTa checkpoint, parameter tensors are weights and biases.
    # position_ids is a registered buffer, not a learnable parameter.
    buffers = {"plm.embeddings.position_ids"}
    parameters = {}
    for name, tensor in state.items():
        if name in buffers:
            continue
        if not (name.endswith(".weight") or name.endswith(".bias")):
            raise ValueError(f"Unrecognized tensor; verify whether it is a parameter: {name}")
        parameters[name] = tensor.numel()

    backbone = sum(count for name, count in parameters.items() if name.startswith("plm."))
    pooler = sum(count for name, count in parameters.items() if name.startswith("pooler."))
    total = sum(parameters.values())
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backbone DeBERTa: {backbone:,}")
    print(f"Pooler: {pooler:,}")
    print(f"Classifiers + discriminators: {total - backbone - pooler:,}")
    print(f"Total parameters: {total:,} ({total / 1e6:.2f} million)")
    # state_dict does not preserve requires_grad; do not infer it from checkpoint.


if __name__ == "__main__":
    main()
