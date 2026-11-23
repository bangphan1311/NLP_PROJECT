"""Compute VUA All POS metrics from the existing best-dev checkpoint."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def metrics(gold, pred):
    if len(gold) != len(pred):
        raise ValueError('Gold/prediction lengths differ')
    c = Counter(zip(gold, pred))
    tn, fp, fn, tp = [c[k] for k in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    return dict(n=len(gold), precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                accuracy=(tp+tn)/len(gold) if gold else 0)


def pos_table(targets, gold, pred):
    rows = []
    for pos in ['ALL', 'VERB', 'ADJ', 'ADV', 'NOUN']:
        indices = [i for i,t in enumerate(targets) if pos == 'ALL' or t['pos'] == pos]
        m = metrics([gold[i] for i in indices], [pred[i] for i in indices])
        rows.append(dict(POS=pos, n=m['n'], **{k:round(100*m[k], 2)
                    for k in ['precision', 'recall', 'f1', 'accuracy']}))
    return rows


def evaluate(work, output):
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_FLAX', '0')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import numpy as np
    import transformers
    from torch.utils.data import DataLoader
    from yacs.config import CfgNode

    work, output = work.resolve(), output.resolve()
    if work == output or work in output.parents:
        raise ValueError('Save CSV outside the original training workspace')
    if output.exists():
        raise FileExistsError(output)
    run = work/'run'
    result = json.loads((run/'result.json').read_text())
    identity = json.loads((run/'identity.json').read_text())
    history = json.loads((run/'history.json').read_text())
    best = max(history, key=lambda r:r['md_dev']['f1'])
    if not result['training_complete'] or best['epoch'] != result['best_epoch']:
        raise ValueError('Attach the final output with the best-dev checkpoint')
    checkpoint = run/f"epochs/epoch_{best['epoch']:02d}.pt"
    for name, version in [('torch',torch.__version__), ('numpy',np.__version__),
                          ('transformers',transformers.__version__)]:
        if version != identity['runtime'][name]:
            raise ValueError(f'Use {name}=={identity["runtime"][name]}')
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU T4')
    test_path = work/'data/data/VUA_All/test.tsv'
    if digest(test_path) != identity['data']['VUA_All/test.tsv']:
        raise ValueError('Test TSV changed')
    if digest(run/'features.pt') != identity['features_sha256']:
        raise ValueError('Cached features changed')
    with test_path.open(encoding='utf-8-sig', newline='') as stream:
        targets = list(csv.DictReader(stream, delimiter='\t'))
    gold = [int(t['label']) for t in targets]
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['identity'] != identity or state['completed_epochs'] != best['epoch']:
        raise ValueError('Checkpoint does not match the selected run')
    features = torch.load(run/'features.pt', map_location='cpu', weights_only=False)
    examples = features['md_test']
    if len(targets) != 50175 or len(examples) != len(targets):
        raise ValueError('Expected 50,175 aligned test targets')
    if gold != [x[3] for x in examples]:
        raise ValueError('TSV labels do not match cached features')
    del features
    sys.path.insert(0, str(work/'src'))
    from model import DeBERTa_base
    from data_utils import dataset, collate_fn
    args = CfgNode.load_cfg(identity['config'])
    args.DATA.plm = str(work/'deberta-v3-base')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = DeBERTa_base(args)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.cuda().eval()
    pred = []
    loader = DataLoader(dataset(examples), batch_size=64, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids, types, mask, _, _ = [t.cuda() for t in batch]
            pred.extend(model(ids, types, mask, domain_idx=1).argmax(1).cpu().tolist())
            if (i+1) % 50 == 0:
                print(f'{len(pred)}/{len(gold)} targets', flush=True)
    overall = metrics(gold, pred)
    if any(abs(overall[k]-best['md_test'][k]) > 1e-8
           for k in ['precision','recall','f1','accuracy']):
        raise ValueError('Overall metrics differ from training log; CSV not saved')
    rows = pos_table(targets, gold, pred)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Checkpoint: epoch {best["epoch"]}; P/R/F1/Accuracy in %')
    print('POS       n     Precision  Recall   F1       Accuracy')
    for row in rows:
        print(f"{row['POS']:<6} {row['n']:>6}   {row['precision']:>6.2f}   "
              f"{row['recall']:>6.2f}   {row['f1']:>6.2f}   {row['accuracy']:>6.2f}")
    print('Saved: ' + str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='Original AdMulPaper output')
    parser.add_argument('--output', type=Path, required=True, help='New POS CSV file')
    opts = parser.parse_args()
    evaluate(opts.work, opts.output)
