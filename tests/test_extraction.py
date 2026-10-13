"""Extraction orchestration tests: guards that run without numpy/torch installed."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from how2sign_pipeline import extraction
from how2sign_pipeline import artifacts
from how2sign_pipeline.fixtures import build_synthetic_source, build_synthetic_output
from how2sign_pipeline.metadata import inspect_dataset

REPO = os.environ.get('VIPOSLT_REPO')


def ns(**kwargs):
    base = dict(repo_root=REPO, output_root=None, dataset_root=None, features='clip,videomae',
                splits='train,val,test', batch_size=8, limit=None, device='cpu', dry_run=False,
                force=False, resume=False, cache_dir=None)
    base.update(kwargs)
    return SimpleNamespace(**base)


def numpy_available():
    try:
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_dry_run_plans_without_reading_data(self):
        repo = self.root / 'repo'
        repo.mkdir()
        plan = extraction.extract(ns(repo_root=str(repo), output_root=str(self.root / 'out'),
                                    dry_run=True, features='clip', splits='train,dev'))
        self.assertTrue(plan['dry_run'])
        self.assertEqual(plan['features'], ['clip'])
        self.assertEqual(plan['splits'], ['train', 'val'])  # dev aliases val

    def test_rejects_unknown_and_duplicate_features(self):
        with self.assertRaises(ValueError):
            extraction.extract(ns(dry_run=True, output_root='x', features='clip,audio'))
        with self.assertRaises(ValueError):
            extraction.extract(ns(dry_run=True, output_root='x', features='clip,clip'))

    def test_rejects_bad_splits_and_batch(self):
        with self.assertRaises(ValueError):
            extraction.extract(ns(dry_run=True, output_root='x', splits='train,holdout'))
        with self.assertRaises(ValueError):
            extraction.extract(ns(dry_run=True, output_root='x', batch_size=0))
        with self.assertRaises(ValueError):
            extraction.extract(ns(dry_run=True, output_root='x', limit=0))

    def test_output_inside_repo_is_rejected(self):
        repo = self.root / 'repo'
        repo.mkdir()
        (repo / 'main.py').write_text('')
        with self.assertRaises(ValueError):
            extraction.extract(ns(repo_root=str(repo), output_root=str(repo / 'gen'), dry_run=True,
                                  features='clip'))


class SourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.report = inspect_dataset(build_synthetic_source(self.root / 'how2sign', per_split=2),
                                      splits=['train', 'val', 'test'])
        self.assertTrue(self.report['samples'])

    def test_unchanged_source_passes(self):
        extraction.assert_source_unchanged(self.report['samples'])

    def test_tampered_source_fails(self):
        sample = self.report['samples'][0]
        Path(sample['folder']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            extraction.assert_source_unchanged(self.report['samples'])

    def test_unsafe_id_fails(self):
        sample = dict(self.report['samples'][0], fileid='../escape')
        with self.assertRaises(ValueError):
            extraction.assert_source_unchanged([sample])

    def test_missing_source_fails(self):
        sample = self.report['samples'][0]
        Path(sample['folder']).unlink()
        with self.assertRaises(ValueError):
            extraction.assert_source_unchanged(self.report['samples'])


class OfflineGuardTests(unittest.TestCase):
    """Non-dry extraction must refuse synthetic data and missing dependencies."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def _prepared(self, synthetic):
        out = self.root / ('syn' if synthetic else 'real')
        source = self.root / 'how2sign'
        build_synthetic_source(source, per_split=2)
        report = inspect_dataset(source, splits=['train', 'val', 'test'])
        artifacts.prepare(report, out, synthetic=synthetic)
        return out

    def test_synthetic_output_refused(self):
        out = self._prepared(synthetic=True)
        with self.assertRaises(ValueError):
            extraction.extract(ns(output_root=str(out), dataset_root=str(self.root / 'how2sign')))
        self.assertFalse((Path(str(out)) / 'features').exists())

    def test_missing_manifest_refused(self):
        with self.assertRaises(ValueError):
            extraction.extract(ns(output_root=str(self.root / 'nope'), dataset_root=str(self.root)))

    @unittest.skipIf(numpy_available(), 'numpy present; dependency-guard path not exercised')
    def test_missing_dependencies_raises_offline(self):
        out = self._prepared(synthetic=False)
        os.environ.pop('HF_HUB_OFFLINE', None)
        with self.assertRaises(RuntimeError) as ctx:
            extraction.extract(ns(output_root=str(out), dataset_root=str(self.root / 'how2sign'),
                                  features='clip', splits='train'))
        self.assertIn('dependencies', str(ctx.exception))
        self.assertEqual(os.environ.get('HF_HUB_OFFLINE'), '1')
        self.assertEqual(os.environ.get('TRANSFORMERS_OFFLINE'), '1')

    def test_dependency_versions_shape(self):
        versions = extraction.dependency_versions()
        self.assertEqual(set(versions), {'torch', 'transformers', 'numpy', 'av', 'Pillow'})
        self.assertTrue(all(isinstance(value, str) for value in versions.values()))


@unittest.skipUnless(numpy_available(), 'checked_resume requires numpy')
class ResumeFactoryTests(unittest.TestCase):
    def test_resume_quarantines_corrupt_batches(self):
        import numpy as np
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        module = SimpleNamespace(get_batch_checkpoint_path=lambda d, i: str(root / f'batch_{i}.npy'))
        np.save(root / 'batch_0.npy', np.zeros((2, 4), dtype=np.float32))
        np.save(root / 'batch_1.npy', np.zeros((0, 4), dtype=np.float32))  # empty -> invalid
        resume = extraction.checked_resume_factory(module, 4)
        self.assertEqual(resume(str(root), 3), 1)
        self.assertTrue(any('invalid' in p.name for p in root.iterdir()))
        self.assertTrue((root / 'batch_1.npy').exists() is False)


if __name__ == '__main__':
    unittest.main()
