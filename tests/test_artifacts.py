"""Artifact preparation/validation tests (standard library only)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from how2sign_pipeline import artifacts
from how2sign_pipeline.fixtures import build_synthetic_source
from how2sign_pipeline.metadata import inspect_dataset
from how2sign_pipeline.npyio import write_float


def make_report(root, splits=('train', 'val', 'test'), per_split=3):
    source = build_synthetic_source(Path(root) / 'how2sign', splits=splits, per_split=per_split)
    report = inspect_dataset(source, splits=list(splits))
    assert not report['errors'], report['errors']
    return report


def add_features(output, report, run):
    for feature, dirname, width_key, enabled in artifacts.FEATURES:
        if not run.get(enabled):
            continue
        info = json.loads((output / 'configs' / 'How2Sign.yaml').read_text())
        base = Path(info['sp_path'] if feature == 'clip' else info['mo_path'])
        for split in ('train', 'val', 'test'):
            mode = artifacts.SPLIT_TO_MODE[split]
            ids = [s['fileid'] for s in report['samples'] if s['source_split'] == split]
            for offset, fileid in enumerate(ids):
                target = base / mode / f'{fileid}.npy'
                target.parent.mkdir(parents=True, exist_ok=True)
                write_float(target, rows=2 + offset, width=run[width_key])


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_prepare_and_validate_roundtrip(self):
        report = make_report(self.root)
        out = self.root / 'out'
        summary = artifacts.prepare(report, out, synthetic=False)
        self.assertEqual(summary['counts'], {'train': 3, 'val': 3, 'test': 3})
        self.assertEqual(summary['modes'], {'train': 'train', 'val': 'dev', 'test': 'test'})
        run = json.loads((out / 'configs' / 'how2sign_edm.yaml').read_text())
        add_features(out, report, run)
        result = artifacts.validate(out, strict=True)
        self.assertTrue(result['ok'], result['errors'])
        self.assertEqual(set(result['splits']), {'train', 'dev', 'test'})

    def test_prepare_rejects_inspection_errors(self):
        report = make_report(self.root)
        report['errors'] = ['boom']
        with self.assertRaises(ValueError):
            artifacts.prepare(report, self.root / 'out')

    def test_prepare_rejects_output_root_inside_repofree_raw(self):
        report = make_report(self.root)
        raw = report['dataset_root']
        with self.assertRaises(ValueError):
            artifacts.prepare(report, Path(raw) / 'nested')

    def test_prepare_is_resumable_and_force_gated(self):
        report = make_report(self.root)
        out = self.root / 'out'
        artifacts.prepare(report, out)
        second = artifacts.prepare(report, out)  # identical -> reused, no error
        self.assertEqual(second['written']['written'], 0)
        target = out / 'configs' / 'How2Sign.yaml'
        target.write_text('{"tampered": true}\n', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            artifacts.prepare(report, out)
        artifacts.prepare(report, out, force=True)
        data = json.loads(target.read_text())
        self.assertIn('dataset_root', data)

    def test_configs_are_portable_and_complete(self):
        report = make_report(self.root)
        out = self.root / 'out'
        artifacts.prepare(report, out)
        info = json.loads((out / 'configs' / 'How2Sign.yaml').read_text())
        for key in artifacts.REQUIRED_INFO_KEYS:
            self.assertIn(key, info)
        raw = str(Path(report['dataset_root']).resolve())
        for key in ('dict_path', 'feature_dir', 'sp_path', 'mo_path', 'phm_path', 'pose_path',
                    'pose_encoder_config_path'):
            self.assertTrue(str(info[key]).startswith(str(out.resolve())), (key, info[key]))
        self.assertEqual(info['dataset_root'], raw)


class ValidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.report = make_report(self.root)
        self.out = self.root / 'out'
        artifacts.prepare(self.report, self.out, synthetic=False)
        self.run = json.loads((self.out / 'configs' / 'how2sign_edm.yaml').read_text())
        add_features(self.out, self.report, self.run)

    def test_missing_feature_detected(self):
        info = json.loads((self.out / 'configs' / 'How2Sign.yaml').read_text())
        victim = next((Path(info['sp_path']) / 'train').glob('*.npy'))
        victim.unlink()
        result = artifacts.validate(self.out, strict=False)
        self.assertFalse(result['ok'])
        self.assertTrue(any('missing' in e for e in result['errors']))

    def test_unexpected_feature_detected(self):
        info = json.loads((self.out / 'configs' / 'How2Sign.yaml').read_text())
        write_float(Path(info['sp_path']) / 'train' / 'not-in-manifest.npy', 2, self.run['sp_hidden_size'])
        result = artifacts.validate(self.out, strict=False)
        self.assertTrue(any('unexpected' in e for e in result['errors']))

    def test_wrong_width_detected(self):
        info = json.loads((self.out / 'configs' / 'How2Sign.yaml').read_text())
        victim = next((Path(info['sp_path']) / 'train').glob('*.npy'))
        write_float(victim, 2, self.run['sp_hidden_size'] + 1)
        result = artifacts.validate(self.out, strict=False)
        self.assertTrue(any('last dim' in e for e in result['errors']))

    def test_target_and_gloss_tampering_detected(self):
        side = self.out / 'preprocess' / 'How2Sign' / 'train_info_ml.json'
        data = json.loads(side.read_text())
        data['0']['gloss'] = 'FAKE'
        data['0']['text'] = ''
        side.write_text(json.dumps(data), encoding='utf-8')
        result = artifacts.validate(self.out, strict=False)
        self.assertTrue(any('gloss must be empty' in e for e in result['errors']))
        self.assertTrue(any('nonempty English target' in e for e in result['errors']))

    def test_strict_raises_on_error(self):
        (self.out / 'preprocess' / 'How2Sign' / 'gloss_dict.npy').unlink()
        with self.assertRaises(ValueError):
            artifacts.validate(self.out, strict=True)

    def test_val_maps_to_dev_only(self):
        dev = json.loads((self.out / 'preprocess' / 'How2Sign' / 'dev_info_ml.json').read_text())
        dev_ids = [r['fileid'] for k, r in dev.items() if str(k).isdigit()]
        val_ids = [s['fileid'] for s in self.report['samples'] if s['source_split'] == 'val']
        self.assertEqual(dev_ids, val_ids)
        self.assertFalse((self.out / 'preprocess' / 'How2Sign' / 'val_info_ml.json').exists())


class SyntheticTests(unittest.TestCase):
    def test_synthetic_validation_skips_features(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'synthetic'
            from how2sign_pipeline.fixtures import build_synthetic_output
            build_synthetic_output(out, per_split=2)
            result = artifacts.validate(out, strict=True)
            self.assertTrue(result['ok'])
            self.assertTrue(result['synthetic'])


if __name__ == '__main__':
    unittest.main()
