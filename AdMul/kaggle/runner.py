"""VUA All, author model/processors/sampler/train/val, full epoch continuation."""
import os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
import tokenizers
import sklearn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, ConcatDataset

from checkpointing import (atomic_json, atomic_save, rng_state, restore_rng,
                           latest_checkpoint, load_checkpoint, save_epoch)
from configs.default import get_config
from model import DeBERTa_base
from data_process import WSD_Processor, MD_Processor
from data_utils import dataset, collate_fn, BalanceSampler
from train_val import train, val, set_random_seeds
from utils import Lamda


class SessionDeadline(Exception):
    pass


class TimedLoader:
    """Keep author batch order; return normally from the notebook before timeout."""
    def __init__(self, loader, deadline):
        self.loader, self.deadline = loader, deadline

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if time.monotonic() >= self.deadline:
                raise SessionDeadline()
            yield batch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def tree_hashes(root, pattern):
    return {str(p.relative_to(root)).replace('\\', '/'): digest(p)
            for p in sorted(Path(root).rglob(pattern)) if p.is_file()}


def runtime():
    return dict(python=platform.python_version(), torch=torch.__version__,
                transformers=transformers.__version__, numpy=np.__version__,
                tokenizers=tokenizers.__version__, sklearn=sklearn.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0), devices=torch.cuda.device_count(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--max-session-epochs', type=int, default=3)
    parser.add_argument('--seconds-left', type=float, default=10.5 * 3600)
    opts = parser.parse_args()
    if opts.max_session_epochs < 1 or opts.seconds_left <= 0:
        parser.error('No session time left or invalid epoch limit')
    work = opts.work.resolve()
    os.chdir(work)
    run = work / 'run'
    run.mkdir(exist_ok=True)
    checkpoint = latest_checkpoint(run)
    deadline = time.monotonic() + opts.seconds_left
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU; TPU is not supported by this CUDA code')
    if transformers.__version__ != '4.18.0' or np.__version__ != '1.21.3':
        raise RuntimeError('Use the isolated Python environment installed by the notebook')
    args = get_config(SimpleNamespace(cfg=str(work / 'src/configs/vua_all.yaml'),
                                      gpu='0', seed=opts.seed, eval=False, task='all'))
    args.defrost()
    args.DATA.data_dir = str(work / 'data/data')
    args.DATA.plm = str(work / 'deberta-v3-base')
    args.TRAIN.output = str(run)
    args.freeze()
    set_random_seeds(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    (run / 'config.yaml').write_text(args.dump(), encoding='utf-8')

    def emit(event, **values):
        value = dict(event=event, **values)
        line = json.dumps(value, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with (run / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    data_root = Path(args.DATA.data_dir)
    data_hashes = tree_hashes(data_root, '*.tsv')
    expected = {'VUA_All/train.tsv': 116622, 'VUA_All/val.tsv': 38628,
                'VUA_All/test.tsv': 50175, 'semcor/threshold/train_3.tsv': 130808,
                'semcor/threshold/test_3.tsv': 4081}
    for name, count in expected.items():
        with (data_root / name).open(encoding='utf-8-sig', newline='') as stream:
            actual = sum(1 for _ in csv.DictReader(stream, delimiter='\t'))
        if actual != count:
            raise ValueError('%s: expected %s rows, got %s' % (name, count, actual))
    cache_path = run / 'features.pt'
    input_identity = dict(data=data_hashes, source=tree_hashes(work / 'src', '*.py'),
                          config=args.dump(), runtime=runtime(),
                          backbone=tree_hashes(work / 'deberta-v3-base', '*'))
    if (run / 'identity.json').exists():
        previous_identity = json.loads((run / 'identity.json').read_text(encoding='utf-8'))
        for key in input_identity:
            if previous_identity[key] != input_identity[key]:
                raise ValueError('Resume rejected: changed ' + key)
        if not cache_path.exists() or previous_identity['features_sha256'] != digest(cache_path):
            raise ValueError('Missing or modified tokenized dataset snapshot')
    elif checkpoint is not None:
        raise ValueError('Checkpoint present without identity.json')

    if cache_path.exists():
        features = torch.load(cache_path, map_location='cpu', weights_only=False)
        if features['input_identity'] != input_identity:
            raise ValueError('Tokenized dataset cache belongs to a different run')
        # Also replay preprocessing RNG on a fresh run with cached features.
        restore_rng(features['before_model_rng'])
    else:
        # Preserve main_vua.py processing/truncation/shuffle order exactly.
        wp, mp = WSD_Processor(args), MD_Processor(args)
        md_train = mp.get_all_train()
        md_dev = mp.get_all_val()
        md_test = mp.get_all_test()
        wsd_train = wp.get_wsd_train()
        if len(wsd_train) >= len(md_train):
            wsd_train = wsd_train[:len(md_train)]
        random.shuffle(wsd_train)
        wsd_test = wp.get_wsd_test()
        features = dict(md_train=md_train, md_dev=md_dev, md_test=md_test,
                        wsd_train=wsd_train, wsd_test=wsd_test,
                        input_identity=input_identity, before_model_rng=rng_state())
        atomic_save(cache_path, features)
        mp.tokenizer.save_pretrained(run / 'tokenizer')
    train_set = ConcatDataset([dataset(features['wsd_train']), dataset(features['md_train'])])
    train_loader = DataLoader(train_set, sampler=BalanceSampler(train_set, args.TRAIN.train_batch_size),
                              batch_size=args.TRAIN.train_batch_size, collate_fn=collate_fn, shuffle=False)
    eval_loaders = {name: DataLoader(dataset(features[key]), batch_size=args.TRAIN.val_batch_size,
                                     collate_fn=collate_fn, shuffle=False)
                    for name, key in [('bsd_test', 'wsd_test'), ('md_dev', 'md_dev'), ('md_test', 'md_test')]}
    sizes = {name: len(features[name]) for name in ('md_train', 'md_dev', 'md_test', 'wsd_train', 'wsd_test')}
    del features
    total_steps = len(train_loader) * args.TRAIN.train_epochs
    identity = dict(input_identity, features_sha256=digest(cache_path),
                    steps_per_epoch=len(train_loader), total_steps=total_steps,
                    gradient_checkpointing=True, mixed_precision=False)
    atomic_json(run / 'identity.json', identity)
    model = DeBERTa_base(args).cuda()
    # HF 4.18 uses PyTorch checkpoint() with preserve_rng_state=True by default.
    # Only recompute activations; retain original FP32 batch and optimizer updates.
    model.plm.gradient_checkpointing_enable()
    if not model.plm.is_gradient_checkpointing:
        raise RuntimeError('Backbone did not enable gradient checkpointing')
    optimizer = AdamW(model.parameters(), lr=args.TRAIN.lr, weight_decay=args.TRAIN.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.TRAIN.warmup_epochs * len(train_loader)),
        num_training_steps=total_steps)
    loss_wsd = torch.nn.CrossEntropyLoss()
    loss_md = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., args.TRAIN.md_class_weight]).cuda())
    loss_task = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., args.TRAIN.wsd_class_weight]).cuda())
    lam = Lamda(args.TRAIN.lambda_lo, args.TRAIN.lambda_hi)
    history, completed, best_epoch, best_f1 = [], 0, None, -1.0
    if checkpoint is not None:
        state = load_checkpoint(checkpoint, identity)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, completed = state['history'], state['completed_epochs']
        best_epoch, best_f1 = state['best_epoch'], state['best_f1']
        if scheduler.last_epoch != completed * len(train_loader):
            raise ValueError('Scheduler progress changed')
        restore_rng(state['rng'])
        del state
    emit('resume' if checkpoint else 'fresh_start', completed_epochs=completed,
         next_epoch=completed + 1, optimizer_restored=checkpoint is not None,
         best_epoch=best_epoch, best_dev_f1=best_f1, sizes=sizes,
         steps_per_epoch=len(train_loader), lr=optimizer.param_groups[0]['lr'], environment=runtime())
    session_epochs = 0
    for epoch in range(completed, args.TRAIN.train_epochs):
        estimated = max((record['seconds'] for record in history), default=3 * 3600)
        if (session_epochs >= opts.max_session_epochs or
                time.monotonic() + estimated * 1.2 + 300 >= deadline):
            emit('session_stop', completed_epochs=completed, next_epoch=completed + 1,
                 reason='Leave time for a complete epoch and checkpoint')
            break
        # Every completed epoch is retained; reserve enough room for the next full save.
        required = 3 * sum(t.numel() * t.element_size() for t in model.state_dict().values()) + 512 * 1024**2
        if shutil.disk_usage(work).free < required:
            raise RuntimeError('Not enough free disk for the next full epoch checkpoint')
        started = time.monotonic()
        emit('epoch_start', epoch=epoch + 1)
        try:
            train(epoch, model, loss_md, loss_wsd, loss_task, optimizer,
                  TimedLoader(train_loader, deadline), total_steps, lam, args, scheduler=scheduler)
            scores = {}
            # Keep ALL original evaluations: DataLoader iterators consume torch RNG,
            # so removing the two test passes changes future training batch order.
            for name, domain in [('md_dev', 1), ('md_test', 1), ('bsd_test', 0)]:
                values = val(model, TimedLoader(eval_loaders[name], deadline), domain_idx=domain)
                scores[name] = dict(zip(('accuracy', 'precision', 'recall', 'f1'), map(float, values)))
        except SessionDeadline:
            # No partial state replaces a completed epoch. Exit normally so Output persists.
            emit('session_stop', completed_epochs=completed, next_epoch=completed + 1,
                 reason='Time budget reached; replay unfinished epoch from last checkpoint')
            break
        torch.cuda.synchronize()
        record = dict(epoch=epoch + 1, seconds=time.monotonic() - started, **scores,
                      lr=optimizer.param_groups[0]['lr'])
        if scores['md_dev']['f1'] > best_f1:
            best_f1, best_epoch = scores['md_dev']['f1'], epoch + 1
        history.append(record)
        path = save_epoch(run, epoch + 1, model, optimizer, scheduler, history, identity, best_epoch, best_f1)
        completed, session_epochs = epoch + 1, session_epochs + 1
        emit('checkpoint_saved', epoch=completed, path=str(path), metrics=record)
    result = dict(training_complete=completed == args.TRAIN.train_epochs,
                  completed_epochs=completed, next_epoch=completed + 1 if completed < args.TRAIN.train_epochs else None,
                  best_epoch=best_epoch, best_dev_f1=best_f1, session_epochs=session_epochs,
                  best_checkpoint=('epochs/epoch_%02d.pt' % best_epoch) if best_epoch else None,
                  note='Best epoch selected by dev only; test passes retained to match author RNG order')
    if best_epoch is not None:
        result['best_epoch_test'] = history[best_epoch - 1]['md_test']
    atomic_json(run / 'result.json', result)
    emit('complete' if result['training_complete'] else 'session_complete', **result)


if __name__ == '__main__':
    main()
