"""Installer lifecycle tests: apply/link/revert in throwaway git clones.

These exercise the real script through a subprocess so the CLI error handling in
``scripts/install_into_repo.py`` is covered too. They are skipped when ``git`` or
an actual serverViPo-SLT checkout is unavailable.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from how2sign_pipeline.fixtures import build_synthetic_output

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / 'scripts' / 'install_into_repo.py'
ENV_REPO = os.environ.get('VIPOSLT_REPO') or str(Path(__file__).resolve().parents[2] / 'serverViPo-SLT')
GIT = shutil.which('git')
HAVE_REPO = GIT is not None and (Path(ENV_REPO) / 'main.py').is_file()

PATCH_FILES = {'main.py', 'test.py', 'ablation_visual.py', 'dataset/loader_ft.py', 'models/edm_ddp.py'}
ARTIFACTS = ('configs/How2Sign.yaml', 'configs/how2sign_edm.yaml', 'preprocess/How2Sign')


def load_installer():
    spec = importlib.util.spec_from_file_location('install_into_repo', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE_REPO, 'requires git and a serverViPo-SLT checkout (VIPOSLT_REPO)')
class InstallerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.repo = self.root / 'repo'
        subprocess.run([GIT, 'clone', '--quiet', '--no-hardlinks', ENV_REPO, str(self.repo)],
                       check=True, capture_output=True)
        self.out = self.root / 'out'
        build_synthetic_output(self.out, per_split=2)

    def run_install(self, *args, expect=0):
        result = subprocess.run([sys.executable, str(SCRIPT), '--repo-root', str(self.repo), *args],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, expect, f'stdout={result.stdout}\nstderr={result.stderr}')
        return result

    def status(self):
        return subprocess.run([GIT, '-C', str(self.repo), 'status', '--porcelain'],
                              text=True, capture_output=True, check=True).stdout.strip()

    def diff_files(self):
        out = subprocess.run([GIT, '-C', str(self.repo), 'diff', '--name-only'],
                             text=True, capture_output=True, check=True).stdout.split()
        return set(out)

    def marker_path(self):
        return self.repo / '.viposlt_integration.json'

    def test_apply_link_revert_clean(self):
        self.run_install('--output-root', str(self.out), '--apply-patch', '--link')
        self.assertEqual(self.diff_files(), PATCH_FILES)
        self.assertTrue(self.marker_path().is_file())
        for relative in ARTIFACTS:
            self.assertTrue((self.repo / relative).exists(), relative)
        self.assertTrue((self.repo / 'configs' / 'How2Sign.yaml').is_symlink())
        state = json.loads(self.marker_path().read_text())
        self.assertEqual(state['patch_status'], 'applied')
        self.assertEqual({entry['relative'] for entry in state['links']}, set(ARTIFACTS))

        self.run_install('--revert')
        self.assertEqual(self.status(), '')
        self.assertFalse(self.marker_path().exists())
        for relative in ARTIFACTS:
            self.assertFalse((self.repo / relative).exists(), relative)

    def test_dry_run_makes_no_changes(self):
        result = self.run_install('--output-root', str(self.out), '--apply-patch', '--link', '--dry-run')
        self.assertIn('diff --git', result.stdout)
        self.assertEqual(self.status(), '')
        self.assertFalse(self.marker_path().exists())

    def test_revert_without_marker_is_refused(self):
        result = self.run_install('--revert', expect=1)
        self.assertIn('No integration marker', result.stderr)

    def test_dirty_tree_blocks_apply(self):
        (self.repo / 'unstaged.py').write_text('x = 1\n', encoding='utf-8')
        result = self.run_install('--output-root', str(self.out), '--apply-patch', '--link', expect=1)
        self.assertIn('dirty', result.stderr)
        self.assertFalse(self.marker_path().exists())

    def test_preexisting_patch_is_retained_on_revert(self):
        patch = PACKAGE / 'patches' / 'viposlt_integration.patch'
        subprocess.run([GIT, '-C', str(self.repo), 'apply', str(patch)], check=True)
        self.run_install('--output-root', str(self.out), '--link')
        state = json.loads(self.marker_path().read_text())
        self.assertEqual(state['patch_status'], 'none')
        self.run_install('--revert')
        self.assertEqual(self.diff_files(), PATCH_FILES)  # preexisting edit preserved
        for relative in ARTIFACTS:
            self.assertFalse((self.repo / relative).exists(), relative)

    def test_missing_artifact_is_reported(self):
        shutil.rmtree(self.out / 'preprocess')
        result = self.run_install('--output-root', str(self.out), '--link', expect=1)
        self.assertIn('Missing prepared artifact', result.stderr)

    def test_existing_unmanaged_artifact_requires_force_and_is_backed_up(self):
        configs = self.repo / 'configs'
        configs.mkdir(parents=True, exist_ok=True)
        original = '{"owner": "preexisting"}\n'
        (configs / 'How2Sign.yaml').write_text(original, encoding='utf-8')
        result = self.run_install('--output-root', str(self.out), '--link', expect=1)
        self.assertIn('unmanaged', result.stderr)
        self.run_install('--output-root', str(self.out), '--link', '--force')
        state = json.loads(self.marker_path().read_text())
        entry = next(e for e in state['links'] if e['relative'] == 'configs/How2Sign.yaml')
        self.assertTrue((self.repo / 'configs' / 'How2Sign.yaml').is_symlink())
        backup = Path(entry['backup'])
        self.assertTrue(backup.is_file())
        self.assertEqual(backup.read_text(encoding='utf-8'), original)
        self.run_install('--revert')
        self.assertEqual((configs / 'How2Sign.yaml').read_text(encoding='utf-8'), original)

    def test_output_root_inside_repo_is_rejected(self):
        result = self.run_install('--output-root', str(self.repo / 'gen'), '--link', expect=2)
        self.assertIn('outside the repository', result.stderr)

    def test_interrupted_installation_blocks_retry(self):
        import hashlib
        patch_hash = hashlib.sha256((PACKAGE / 'patches' / 'viposlt_integration.patch').read_bytes()).hexdigest()
        self.marker_path().write_text(json.dumps({
            'version': 2, 'installation_id': '0' * 32, 'patch_version': patch_hash,
            'patch_status': 'pending', 'output_root': None,
            'timestamp': '2026-01-01T00:00:00+00:00', 'links': []}), encoding='utf-8')
        result = self.run_install('--output-root', str(self.out), '--apply-patch', '--link', expect=1)
        self.assertIn('interrupted', result.stderr)


class InstallerContractTests(unittest.TestCase):
    @unittest.skipUnless(SCRIPT.is_file(), 'installer script missing')
    def test_patch_allowlist_matches_ownerless_repo(self):
        installer = load_installer()
        self.assertEqual(installer.PATCH_FILES, PATCH_FILES)
        self.assertEqual(tuple(installer.ARTIFACTS), ARTIFACTS)

    def test_artifacts_are_created_by_prepare(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'out'
            build_synthetic_output(out, per_split=1)
            for relative in ARTIFACTS:
                self.assertTrue((out / relative).exists(), relative)


if __name__ == '__main__':
    unittest.main()
