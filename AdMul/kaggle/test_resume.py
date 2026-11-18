"""CPU proof with the author's real sampler, train(), val() and a tiny model."""
import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'kaggle'))
import checkpointing as cp
from checkpointing import *
from data_utils import dataset, collate_fn, BalanceSampler
from train_val import train, val, set_random_seeds
from utils import Lamda


class Config(NS):
    def defrost(self):
        pass

    def freeze(self):
        pass

    def dump(self):
        return json.dumps(vars(self), default=vars, sort_keys=True)


def get_config(opts):
    return Config(seed=opts.seed, DATA=NS(data_dir='', plm=''),
                  TRAIN=NS(output='', train_batch_size=4, val_batch_size=2,
                           lr=3e-3, weight_decay=.01, warmup_epochs=2, train_epochs=8,
                           md_class_weight=4, wsd_class_weight=1, wsd_weight=.2,
                           global_weight=.1, local_weight=.1, lambda_lo=.1, lambda_hi=.8))


class Tiny(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature = torch.nn.Linear(1, 4)
        self.dropout = torch.nn.Dropout(.25)
        self.wsd = torch.nn.Linear(4, 2)
        self.md = torch.nn.Linear(4, 2)
        self.task = torch.nn.Linear(4, 2)
        self.local0 = torch.nn.Linear(4, 2)
        self.local1 = torch.nn.Linear(4, 2)
        self.plm = NS(is_gradient_checkpointing=True, gradient_checkpointing_enable=lambda: None)

    def cuda(self):
        return self

    def forward(self, ids, types, mask, adv_lmbd=.01, domain_idx=1):
        features = self.dropout(self.feature(ids.float()))
        if not self.training:
            return (self.md if domain_idx else self.wsd)(features)
        half = features.shape[0] // 2
        a, b = features[:half], features[half:]
        return self.wsd(a), self.md(b), self.task(features), [self.local0(a), self.local1(a)], [self.local0(b), self.local1(b)]


class Processor:
    def __init__(self, args):
        self.root = Path(args.DATA.data_dir)
        self.tokenizer = NS(save_pretrained=lambda path: None)

    def read(self, name, domain):
        with (self.root / name).open(newline='') as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        return [[[int(row['lemma'])], [0], [1], int(row['label']), domain] for row in rows]

    def get_all_train(self):
        return self.read('VUA_All/train.tsv', 1)

    def get_all_val(self):
        return self.read('VUA_All/val.tsv', 1)

    def get_all_test(self):
        return self.read('VUA_All/test.tsv', 1)

    def get_wsd_train(self):
        return self.read('semcor/threshold/train_3.tsv', 0)

    def get_wsd_test(self):
        return self.read('semcor/threshold/test_3.tsv', 0)


def schedule(optimizer, num_warmup_steps, num_training_steps):
    def factor(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        return max(0., (num_training_steps - step) / max(1, num_training_steps - num_warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_runner():
    source = (ROOT / 'kaggle/runner.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]

    class ToyCounts(ast.NodeTransformer):
        def visit_Constant(self, node):
            # Only shrink the expected dataset row counts for the CPU fixture.
            counts = {116622: 4, 38628: 2, 50175: 2, 130808: 6, 4081: 2}
            if isinstance(node.value, int) and node.value in counts:
                return ast.copy_location(ast.Constant(counts[node.value]), node)
            return node

    tree = ast.fix_missing_locations(ToyCounts().visit(tree))
    env = dict(globals(), SimpleNamespace=NS, transformers=NS(__version__='4.18.0'),
               np=NS(__version__='1.21.3'), DeBERTa_base=Tiny,
               MD_Processor=Processor, WSD_Processor=Processor,
               AdamW=torch.optim.AdamW, get_linear_schedule_with_warmup=schedule)
    exec(compile(tree, str(ROOT / 'kaggle/runner.py'), 'exec'), env)
    env['runtime'] = lambda: {'test': 'CPU', 'optimizer': 'test-only torch AdamW'}
    return env


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='admul-checkpoint-proof-')
        self.root = Path(self.temp.name)
        self.work = self.root / 'AdMulPaper'
        self.previous_cwd = Path.cwd()
        self.env = make_runner()
        self.stack = contextlib.ExitStack()
        # Local torch 2.x removed the parent constructor argument accepted in target torch 1.13.
        # Adapt only the test harness; author sampler source in the notebook stays untouched.
        self.stack.enter_context(patch.object(torch.utils.data.Sampler, '__init__', lambda self, data=None: None))
        self.stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda tensor: tensor))
        for name, function in dict(is_available=lambda: True, manual_seed=lambda seed: None,
                                   manual_seed_all=lambda seed: None, synchronize=lambda: None,
                                   get_rng_state_all=lambda: []).items():
            self.stack.enter_context(patch.object(torch.cuda, name, function))
        self.fresh()

    def tearDown(self):
        os.chdir(self.previous_cwd)
        self.stack.close()
        self.temp.cleanup()

    def fresh(self):
        self.work.mkdir()
        (self.work / 'src').mkdir()
        (self.work / 'src/runner.py').write_text('immutable source snapshot')
        (self.work / 'deberta-v3-base').mkdir()
        (self.work / 'deberta-v3-base/config.json').write_text('{}')
        for name, count in [('VUA_All/train.tsv', 4), ('VUA_All/val.tsv', 2),
                            ('VUA_All/test.tsv', 2), ('semcor/threshold/train_3.tsv', 6),
                            ('semcor/threshold/test_3.tsv', 2)]:
            path = self.work / 'data/data' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('lemma\tlabel\n' + ''.join('%d\t%d\n' % (i+1, i % 2) for i in range(count)))

    def run_session(self, epochs=3, seconds=1000000):
        argv = ['runner.py', '--work', str(self.work), '--max-session-epochs', str(epochs),
                '--seconds-left', str(seconds)]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.env['main']()
        return json.loads((self.work / 'run/result.json').read_text())

    def state(self, epoch):
        return torch.load(self.work / ('run/epochs/epoch_%02d.pt' % epoch), map_location='cpu', weights_only=False)

    def next_kaggle_session(self):
        # Same working path on a new machine, copied from the previous saved Output.
        saved = self.root / 'previous-output'
        shutil.copytree(self.work, saved)
        os.chdir(self.root)
        shutil.rmtree(self.work)
        shutil.copytree(saved, self.work)
        shutil.rmtree(saved)
        random.seed(934)
        np.random.seed(45)
        torch.manual_seed(768)

    def test_all_epochs_resume_matches_continuous(self):
        self.assertEqual(self.run_session()['completed_epochs'], 3)
        first_rng = self.state(3)['rng']
        self.next_kaggle_session()
        self.assertEqual(self.run_session()['completed_epochs'], 6)
        self.next_kaggle_session()
        result = self.run_session()
        self.assertTrue(result['training_complete'])
        split = self.state(8)
        self.assertEqual(len(list((self.work / 'run/epochs').glob('*.pt'))), 8)
        self.assertEqual(split['global_step'], 16)
        self.assertEqual(split['scheduler']['last_epoch'], 16)
        self.assertEqual(len(split['history']), 8)
        best = max(split['history'], key=lambda record: record['md_dev']['f1'])
        self.assertEqual(best['epoch'], result['best_epoch'])
        # Run fresh for 8 uninterrupted epochs and compare real CPU tensors/state.
        os.chdir(self.root)
        shutil.rmtree(self.work)
        self.fresh()
        self.run_session(epochs=8)
        continuous = self.state(8)
        self.assertEqual(split['identity'], continuous['identity'])
        for name, tensor in split['model'].items():
            self.assertTrue(torch.equal(tensor, continuous['model'][name]), name)
        for key, values in split['optimizer']['state'].items():
            for field, tensor in values.items():
                self.assertTrue(torch.equal(tensor, continuous['optimizer']['state'][key][field]), field)
        self.assertEqual(split['scheduler'], continuous['scheduler'])
        self.assertEqual(split['rng']['python'], continuous['rng']['python'])
        np.testing.assert_equal(split['rng']['numpy'], continuous['rng']['numpy'])
        self.assertTrue(torch.equal(split['rng']['torch'], continuous['rng']['torch']))
        self.assertTrue(torch.equal(first_rng['torch'], self.state(3)['rng']['torch']))
        for left, right in zip(split['history'], continuous['history']):
            for key in ['md_dev', 'md_test', 'bsd_test', 'lr', 'epoch']:
                self.assertEqual(left[key], right[key])

    def test_refuses_changed_inputs_and_partial_save(self):
        self.run_session(epochs=1)
        run = self.work / 'run'
        original = cp.latest_checkpoint(run)
        with patch.object(cp.torch, 'save', side_effect=RuntimeError('interrupted save')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                cp.atomic_save(run / 'epochs/epoch_02.pt', {'incomplete': True})
        self.assertEqual(cp.latest_checkpoint(run), original)
        self.assertTrue(original.exists())
        current = self.state(1)
        changed = copy.deepcopy(current['identity'])
        changed['runtime'] = {'changed': True}
        with self.assertRaisesRegex(ValueError, 'runtime'):
            cp.load_checkpoint(original, changed)
        path = self.work / 'data/data/VUA_All/train.tsv'
        path.write_text(path.read_text().replace('1\t0', '9\t0'))
        with self.assertRaisesRegex(ValueError, 'changed data'):
            self.run_session()

    def test_stops_before_epoch_and_discards_partial_epoch(self):
        result = self.run_session(seconds=1)
        self.assertEqual(result['completed_epochs'], 0)
        self.assertFalse(result['training_complete'])
        self.assertIsNone(cp.latest_checkpoint(self.work / 'run'))
        # Once an epoch exists, a deadline interruption must keep its checkpoint.
        self.run_session(epochs=1)
        original = cp.latest_checkpoint(self.work / 'run')
        original_train = self.env['train']

        def interrupted_train(*args, **kwargs):
            original_train(*args, **kwargs)
            raise self.env['SessionDeadline']()

        self.env['train'] = interrupted_train
        result = self.run_session()
        self.assertEqual(result['completed_epochs'], 1)
        self.assertEqual(cp.latest_checkpoint(self.work / 'run'), original)

    def test_notebook_sources_and_cells(self):
        notebook = json.loads((ROOT / 'AdMul_Kaggle_Checkpoints.ipynb').read_text(encoding='utf-8'))
        environment_cells = []
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            compile(source, 'notebook-cell-%s' % i, 'exec')
            if "'pip', 'install', '--python', PYTHON" in source:
                environment_cells.append(source)
                self.assertIn("'six==1.16.0'", source)
                self.assertIn('import six;', source)
                self.assertIn('import torch.utils.checkpoint;', source)
                self.assertIn("str(WORK / 'src/smoke.py')", source)
                self.assertIn('from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer', source)
            if source.startswith('SOURCES = '):
                embedded = ast.literal_eval(ast.parse(source).body[0].value)
                for name in ['model.py', 'data_process.py', 'data_utils.py', 'train_val.py', 'utils.py']:
                    self.assertEqual(embedded[name], (ROOT / name).read_text(encoding='utf-8'))
                for name in ['runner.py', 'checkpointing.py', 'smoke.py']:
                    self.assertEqual(embedded[name], (ROOT / 'kaggle' / name).read_text(encoding='utf-8'))
                    compile(embedded[name], name, 'exec')
                imports = [node for node in ast.parse(embedded['runner.py']).body if isinstance(node, ast.Import)]
                self.assertTrue(any(alias.name == 'torch.utils.checkpoint' for node in imports for alias in node.names))
        self.assertEqual(len(environment_cells), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
