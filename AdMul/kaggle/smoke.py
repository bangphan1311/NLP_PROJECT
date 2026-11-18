"""Exercise the real author model/train loop with tiny random DeBERTa on Kaggle."""
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.utils.checkpoint
from transformers import DebertaV2Config, DebertaV2Model, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from model import DeBERTa_base
from data_utils import dataset, collate_fn
from train_val import train, set_random_seeds
from utils import Lamda


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for the Kaggle startup smoke test')
    set_random_seeds(4)
    with tempfile.TemporaryDirectory(prefix='admul-smoke-') as directory:
        config = DebertaV2Config(vocab_size=32, hidden_size=24, num_hidden_layers=1,
                                 num_attention_heads=4, intermediate_size=48,
                                 max_position_embeddings=32, relative_attention=True,
                                 pos_att_type=['p2c', 'c2p'], position_buckets=8,
                                 max_relative_positions=16, conv_kernel_size=3,
                                 norm_rel_ebd='layer_norm')
        DebertaV2Model(config).save_pretrained(directory)
        args = NS(DATA=NS(plm=directory),
                  MODEL=NS(dropout=0., embed_dim=24, num_classes=2, num_tasks=2),
                  TRAIN=NS(wsd_weight=.2, global_weight=.1, local_weight=.1))
        model = DeBERTa_base(args).cuda()
        model.plm.gradient_checkpointing_enable()
        optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=2)
        examples = [[[1, 2, 3, 4, 5, 6, 7, 8], [0]*8, [1]*8, i % 2, i // 2] for i in range(4)]
        loader = DataLoader(dataset(examples), batch_size=4, collate_fn=collate_fn)
        losses = [torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).cuda())
                  for weight in ([1., 4.], [1., 1.], [1., 1.])]
        train(0, model, *losses, optimizer, loader, 2, Lamda(.1, .8), args, scheduler=scheduler)
        if not optimizer.state or not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError('Tiny checkpointed training smoke test failed')
        model.eval()
        batch = next(iter(loader))
        with torch.no_grad():
            logits = model(*(value.cuda() for value in batch[:3]), domain_idx=1)
        if tuple(logits.shape) != (4, 2) or not torch.isfinite(logits).all().item():
            raise RuntimeError('Tiny evaluation smoke test failed')
    torch.cuda.empty_cache()
    print('SMOKE OK: author model, gradient checkpointing, backward, HF AdamW and evaluation', flush=True)


if __name__ == '__main__':
    main()
