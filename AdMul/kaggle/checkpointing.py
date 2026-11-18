"""Epoch checkpoints. Only load these pickle files from your own trusted runs."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed from checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def latest_checkpoint(run_dir):
    # A final .pt only appears after a complete, flushed save; ignore unfinished .tmp.
    # Scan files so a crash after saving .pt but before latest.json also recovers.
    paths = sorted((Path(run_dir) / 'epochs').glob('epoch_*.pt'))
    return paths[-1] if paths else None


def load_checkpoint(path, identity):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != 'admul-epoch-v1':
        raise ValueError('Not a full checkpoint from this notebook')
    if state['identity'] != identity:
        changed = [key for key in identity if state['identity'].get(key) != identity[key]]
        raise ValueError('Resume rejected: changed ' + ', '.join(changed))
    epoch = state['completed_epochs']
    if epoch != len(state['history']) or state['global_step'] != epoch * identity['steps_per_epoch']:
        raise ValueError('Checkpoint epoch/history/step mismatch')
    if state['scheduler']['last_epoch'] != state['global_step']:
        raise ValueError('Checkpoint scheduler progress mismatch')
    return state


def save_epoch(run_dir, epoch, model, optimizer, scheduler, history, identity, best_epoch, best_f1):
    run_dir = Path(run_dir)
    if epoch != len(history):
        raise ValueError('Save only after a complete epoch and its evaluations')
    path = run_dir / 'epochs' / ('epoch_%02d.pt' % epoch)
    if path.exists():
        raise FileExistsError('Refusing to overwrite ' + str(path))
    payload = dict(format='admul-epoch-v1', completed_epochs=epoch,
                   global_step=epoch * identity['steps_per_epoch'], identity=identity,
                   model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), rng=rng_state(),
                   best_epoch=best_epoch, best_f1=best_f1, history=history)
    atomic_save(path, payload)
    atomic_json(run_dir / 'history.json', history)
    atomic_json(run_dir / 'latest.json', dict(completed_epochs=epoch, checkpoint=str(path.relative_to(run_dir)),
                                            best_epoch=best_epoch, best_dev_f1=best_f1))
    return path
