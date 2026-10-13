"""CLI tests. Repo-dependent commands are skipped without a real checkout."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from how2sign_pipeline import cli
from how2sign_pipeline.fixtures import build_synthetic_source

REPO = os.environ.get('VIPOSLT_REPO')
HAVE_REPO = bool(REPO) and (Path(REPO) / 'main.py').is_file()


class ParserTests(unittest.TestCase):
    def test_parser_has_all_subcommands(self):
        parser = cli.build_parser()
        subcommands = set()
        for action in parser._actions:
            if getattr(action, 'choices', None) and 'inspect' in action.choices:
                subcommands = set(action.choices)
        self.assertEqual(subcommands, {'inspect', 'prepare', 'extract', 'validate', 'all', 'smoke-test'})

    def test_version_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(['--version'])
        self.assertEqual(ctx.exception.code, 0)

    def test_unknown_command_is_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(['bogus'])
        self.assertEqual(ctx.exception.code, 2)


class InspectTests(unittest.TestCase):
    def test_inspect_synthetic_reports_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = build_synthetic_source(root / 'how2sign', per_split=2)
            report = root / 'report.json'
            code = cli.main(['inspect', '--dataset-root', str(source), '--report', str(report)])
            self.assertEqual(code, 0)
            self.assertTrue(report.is_file())
            data = json.loads(report.read_text())
            self.assertEqual(set(data['split_counts']), {'train', 'val', 'test'})
            self.assertEqual(len(data['samples']), 6)


@unittest.skipUnless(HAVE_REPO, 'requires VIPOSLT_REPO checkout')
class RepoCommandTests(unittest.TestCase):
    def test_smoke_test_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'smoke'
            code = cli.main(['smoke-test', '--repo-root', REPO, '--output-root', str(out)])
            self.assertEqual(code, 0)
            self.assertTrue((out / 'configs' / 'How2Sign.yaml').is_file())
            self.assertTrue((out / 'reports' / 'validation_report.json').is_file())

    def test_prepare_then_validate_manifests_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = build_synthetic_source(root / 'how2sign', per_split=2)
            out = root / 'out'
            code = cli.main(['prepare', '--repo-root', REPO, '--dataset-root', str(source),
                             '--output-root', str(out), '--allow-sentence-clips'])
            self.assertEqual(code, 0)
            code = cli.main(['validate', '--repo-root', REPO, '--output-root', str(out),
                             '--no-features'])
            self.assertEqual(code, 0)

    def test_validate_missing_output_returns_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            code = cli.main(['validate', '--repo-root', REPO, '--output-root', str(Path(temp) / 'nope')])
            self.assertEqual(code, 1)

    def test_prepare_requires_dataset_root(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(['prepare', '--repo-root', REPO, '--output-root', '/tmp/whatever'])
        self.assertEqual(ctx.exception.code, 2)

    def test_extract_dry_run_reports_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'out'
            code = cli.main(['extract', '--repo-root', REPO, '--output-root', str(out),
                             '--features', 'clip', '--splits', 'train', '--dry-run',
                             '--dataset-root', str(Path(temp) / 'raw')])
            self.assertEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
