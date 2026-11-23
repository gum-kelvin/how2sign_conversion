# AUDIT — How2Sign -> ViPo-SLT pipeline (Hardened v2)

Date: 2026-10-12. Evaluator environment: Python 3.11.2, no `numpy`, `torch`,
`transformers`, `pytest`, `yaml`, `PIL` or `cv2`; `git` present. Everything below
was run in this environment unless labelled `[real-data-tested]` / `[GPU-tested]`.

> Note: the handover claimed Python 3.14.7 + numpy. The actual offline workspace
> is Python 3.11.2 with **no** numpy, so `numpy`-dependent code paths are guarded
> and exercised only through stdlib mocks (`skipped=1` for the numpy resume test).

## 1. Verdict

Gate A (offline) passes. Gate B (real data + GPU) is **PENDING** and must be run
on the user's server; exact commands are in section 6.

## 2. Acceptance gate summary

| # | Gate A criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | External package; only patch + installer artifacts in repo; `--dry-run`/`--revert` isolated | PASS `[stdlib-fixture-tested]` | `tests/test_installer.py` (temp git clones) |
| 2 | `configs/How2Sign.yaml` (all keys, portable) + `configs/how2sign_edm.yaml` | PASS `[static-checked]` | `how2sign_eval_static.py`, `tests/test_artifacts.py` |
| 3 | How2Sign split branch; PHOENIX/CSL unchanged | PASS `[static-checked]` | `how2sign_eval_static.py`, `tests/test_guards.py` |
| 4 | Manifest-driven discovery; no `fname.split(".")[0]`; PHOENIX regression | PASS | `how2sign_eval_static.py`, `tests/test_contracts.py`, `tests/test_guards.py` |
| 5 | `dict_path`/`pose` loads conditional or valid placeholders; no fake gloss | PASS | `how2sign_eval_static.py`, `tests/test_guards.py` |
| 6 | `compileall` + stdlib suite | PASS `[compiled]` `[stdlib-fixture-tested]` | 73 tests, 1 skipped; static check (a) |
| 7 | Smoke test exercises a real `PrecomputedFeatureDataset` batch | PASS `[stdlib-fixture-tested]` | `smoke-test` (finite tensors, ids, padded lengths) |
| 8 | Anti-leak regression (target absent from prompt) | PASS `[static-checked]` | `how2sign_eval_static.py`, `tests/test_guards.py` |
| 9 | Final report (this file) | PASS | — |

## 3. What was wrong in the delivered package (evaluation)

Two confirmed defects, both fixed here:

1. `scripts/install_into_repo.py` — `PATCH_FILES` allowlist named files that do
   not exist (`models/visual_backbone.py`, `utils.py`), so every install aborted
   with *"Patch contains unexpected repository files"*. The real patch touches
   `main.py`, `test.py`, `ablation_visual.py`, `dataset/loader_ft.py`,
   `models/edm_ddp.py`. Fixed; the dirty-tree guard now only blocks when a patch
   actually needs applying (the live repo may already carry the uncommitted
   integration).
2. `scripts/how2sign_eval_static.py` — the prompt checker rejected a *bound*
   variable (`_`) in `[self.prompt for _ in range(bs)]`, and the `/Users/`
   substring rule false-positived on generated Windows paths. Fixed: only
   `ast.Load` names are counted, and path portability uses a root-relative check.

Additional defect found and fixed while writing tests:

3. `artifacts.prepare` was not actually resumable — `source_manifest.json`
   embedded a timestamp, so re-running `prepare` raised `FileExistsError`
   (`resume` was accepted but unused). Fixed: the fingerprint covers a
   timestamp-free source identity; identical re-runs reuse artifacts, differing
   content still requires `--force`, and `resume=False` rewrites identical
   artifacts.

## 4. Files

External package (`how2sign_pipeline/`): new/regenerated
`artifacts.py`, `cli.py`, `__main__.py`, `fixtures.py`, `npyio.py`,
`README.md`, `pyproject.toml`, `AUDIT.md`,
`tests/test_artifacts.py`, `tests/test_cli.py`, `tests/test_extraction.py`,
`tests/test_installer.py`; fixed `scripts/install_into_repo.py`,
`scripts/how2sign_eval_static.py`.

Repository (`patches/viposlt_integration.patch`, 95 insertions / 48 deletions):

```
main.py              | 22 ++++++++++--
test.py              | 20 +++++++++--
ablation_visual.py   | 15 +++++++--
dataset/loader_ft.py | 85 +++++++++++++++++++++++-------------------
models/edm_ddp.py    |  1 +
```

The patch applies cleanly to the unmodified checkout and, when applied, produces
exactly this 5-file diff. No pipeline source is copied into the repo.

## 5. Commands actually run

```bash
# Compile every pipeline module and the patched repo modules.
python -m compileall how2sign_pipeline scripts
```

Result: PASS `[compiled]`.

```bash
# Full standard-library suite against a patched checkout.
VIPOSLT_REPO=/path/to/patched-repo python -m unittest discover -s tests -v
```

Result: `Ran 73 tests ... OK (skipped=1)`. The single skip is the numpy-only
resume-quarantine test.

```bash
# Offline static gate against the patched checkout.
python scripts/how2sign_eval_static.py --repo-root /path/to/patched-repo \
  --output-root /path/to/output
```

Result `[static-checked]`: 5/5 PASS (compiled, dataset+run config keys,
manifest discovery + period-safe ids, How2Sign train/dev/test selection,
instruction prompt excludes target variables). Against a clean checkout it
correctly FAILS discovery and split-selection, proving the checks are live.

```bash
# End-to-end offline run: fixture -> prepare -> real loader -> validate.
python -m how2sign_pipeline smoke-test --repo-root /path/to/patched-repo --keep
```

Result `[stdlib-fixture-tested]`: for `train`/`dev`/`test` the real
`PrecomputedFeatureDataset` produced batches with `clip [2,3,2048]`,
`videomae [2,3,1024]`, `targets_match: true`, `padding_verified: true`,
`validation.ok: true`.

## 6. Gate B — PENDING (run on the server with real data + GPU)

Not run here: no How2Sign data, no CUDA, no cached weights. Exact commands and
expected observable results:

### B.10 real extraction smoke (1–2 samples/feature) + loader batch

```bash
python -m how2sign_pipeline extract --repo-root "$VIPOSLT_REPO" \
  --output-root "$VIPOSLT_FEATURE_ROOT" --features clip,videomae \
  --device cuda:0 --limit 2 --splits train --resume
```

Expected: exit 0; `extraction_report.json` lists 2 `clip` and 2 `videomae`
samples with `status: extracted` and shapes `[T,2048]` / `[T,1024]`; no download
is attempted (offline env vars are set by the pipeline). Mark `[real-data-tested]`.

```bash
python -m how2sign_pipeline validate --repo-root "$VIPOSLT_REPO" \
  --output-root "$VIPOSLT_FEATURE_ROOT" --strict
```

Expected: `ok: true`, no missing/unexpected feature files, widths match config.

### B.11 one-batch GPU model/training smoke

```bash
python scripts/model_smoke.py --repo-root "$VIPOSLT_REPO" \
  --output-root "$VIPOSLT_FEATURE_ROOT" --device cuda:0 --batch-size 2
```

Expected: prints `{"status":"PASS","verification":"GPU-tested", "finite_gradients":true,"optimizer_steps":1}`
with a finite loss. Requires cached LLM weights under the HF cache and
`$VIPOSLT_REPO/data/models`. Mark `[GPU-tested]`.

## 7. Modalities

- **Active:** `rgb_front` CLIP spatial (`sp_hidden_size=2048`), VideoMAE motion
  (`mo_hidden_size=1024`).
- **Preserved, unused:** side camera, pose, depth. `include_pose: false`.
- **Gloss/CTC:** disabled; `gloss_dict.npy` is a legitimate empty `{}`.

## 8. Unresolved limitations / risks

- `git` must be available for `--apply-patch`/`--revert`; artifact symlinks fall
  back to copies on platforms without symlink permission (Windows junctions are
  explicitly refused).
- A repo that already contains the uncommitted integration patch is treated as
  *preexisting ownership*: the installer links artifacts and records
  `patch_status: none`, so `--revert` removes the links but deliberately does
  **not** reverse a patch it did not apply (protects the user's edits).
- Real feature widths are asserted against the run config; if real extraction
  measures different widths, update `configs/how2sign_edm.yaml` to the measured
  values and document why (Gate B).
- Pose support for How2Sign is intentionally unimplemented and fails fast.
