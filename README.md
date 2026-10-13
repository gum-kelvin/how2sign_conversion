# how2sign_pipeline

A complete, step by step guide to converting the **How2Sign** dataset into the
exact layout that **ViPo-SLT** (serverViPo-SLT) expects, running the repository's
own CLIP and VideoMAE feature extractors, and validating the result before you
train.

This pipeline is **external**. It does not get copied into ViPo-SLT. The
repository only receives one reversible patch file plus a few small generated
artifacts. Everything else (features, manifests, reports) stays in your output
folder.

It runs on Linux, macOS, Windows, and on a headless GPU server such as
JupyterHub.

> This guide replaces the older short README. Every command below is shown
> together with a short description of what it produces.

---

## Table of contents

1. [What this pipeline does](#1-what-this-pipeline-does)
2. [Requirements before you start](#2-requirements-before-you-start)
3. [The three environment variables](#3-the-three-environment-variables)
4. [Step 0: Get the files onto the server](#4-step-0-get-the-files-onto-the-server)
5. [Step 1: Set the environment variables](#5-step-1-set-the-environment-variables)
6. [Step 2: Sanity check Python and git](#6-step-2-sanity-check-python-and-git)
7. [Step 3: Inspect the dataset (read only)](#7-step-3-inspect-the-dataset-read-only)
8. [Step 4: Prepare configs and manifests](#8-step-4-prepare-configs-and-manifests)
9. [Step 5: Install the integration into the repo](#9-step-5-install-the-integration-into-the-repo)
10. [Step 6: Offline smoke test](#10-step-6-offline-smoke-test)
11. [Step 7: Extract features on the GPU](#11-step-7-extract-features-on-the-gpu)
12. [Step 8: Validate the artifacts](#12-step-8-validate-the-artifacts)
13. [Step 9: Train](#13-step-9-train)
14. [Optional: run everything at once](#14-optional-run-everything-at-once)
15. [Optional: one real GPU model step](#15-optional-one-real-gpu-model-step)
16. [Optional: offline static gate](#16-optional-offline-static-gate)
17. [Command reference](#17-command-reference)
18. [Understanding the output layout](#18-understanding-the-output-layout)
19. [Modalities](#19-modalities)
20. [Resume, force, and safety](#20-resume-force-and-safety)
21. [Troubleshooting](#21-troubleshooting)
22. [Running the test suite](#22-running-the-test-suite)
23. [Uninstalling and reverting](#23-uninstalling-and-reverting)
24. [Known limitations](#24-known-limitations)
25. [Licence and provenance](#25-licence-and-provenance)

---

## 1. What this pipeline does

The pipeline performs five jobs, in order:

1. **Inspect** the How2Sign folder and build a read only inventory. It matches
   each sentence to its frontal RGB video, checks the timestamps, and reports any
   problem instead of guessing.
2. **Prepare** the ViPo-SLT config files, the feature extractor manifests, the
   loader manifests, and a legitimate empty gloss dictionary.
3. **Install** the minimal, reversible ViPo-SLT integration. This applies one
   patch to five repository files and creates a few small artifacts that ViPo-SLT
   looks up by relative path.
4. **Extract** CLIP spatial features and VideoMAE motion features using the
   repository's existing extractor scripts. This is the GPU stage. It is offline
   and resumable.
5. **Validate** that every produced file satisfies the ViPo-SLT loader contract
   (ids match, shapes match, dtypes are finite, English targets are present, no
   glosses were invented).

Nothing in this pipeline downloads data. Model weights must already be in your
Hugging Face cache.

---

## 2. Requirements before you start

You need all of the following on the machine that will do the conversion.

| Requirement | Notes |
| --- | --- |
| Python 3.9 or newer | The pipeline core is standard library only. |
| `git` | Needed to apply and reverse the integration patch. |
| A ViPo-SLT checkout | Your existing serverViPo-SLT repo, without the changedFiles patch. |
| The How2Sign dataset | Extracted on disk, read only. |
| Free disk space | See the storage math below. |

For the **GPU extraction stage** you also need, inside the repository's Python
environment:

| Requirement | Notes |
| --- | --- |
| `numpy` | Already used by the repo. |
| `torch` with CUDA | Already used by the repo. |
| `transformers` | Already used by the repo. |
| `av` (PyAV) | Used to decode video clips. |
| `Pillow` | Used by the CLIP reader. |
| A CUDA GPU | Check with `nvidia-smi`. |
| Cached model weights | `openai/clip-vit-large-patch14` and `MCG-NJU/videomae-large`. |

Storage math, per sample, frontal camera only:

```
feature bytes = 4 bytes per float x (2048 + 1024) floats per frame x frames per clip
```

So a 100 frame clip costs roughly 1.2 MB across the two feature files. Add the
originals, which are never re-encoded. Run the small pilot in Step 7 to measure
your own numbers.

The pipeline itself is invoked with an explicit repository path and never
assumes the current directory is the repository.

---

## 3. The three environment variables

Set these once, and most commands become short. Each command also accepts an
explicit flag if you prefer.

| Variable | Meaning | Example (Linux or server) |
| --- | --- | --- |
| `VIPOSLT_REPO` | Absolute path to the ViPo-SLT checkout. | `/home/jovyan/serverViPo-SLT` |
| `HOW2SIGN_ROOT` | Read only path to the extracted How2Sign dataset. | `/home/jovyan/data/how2sign` |
| `VIPOSLT_FEATURE_ROOT` | Output folder for all generated files. Must be outside the repo. | `/home/jovyan/data/how2sign_features` |

Rules:

- All three must be absolute paths, and `VIPOSLT_FEATURE_ROOT` must sit outside
  `VIPOSLT_REPO`. The pipeline refuses overlapping paths on purpose.
- The pipeline expands environment variables itself, so passing `$HOW2SIGN_ROOT`
  in a shell also works.

---

## 4. Step 0: Get the files onto the server

You have a single archive, `how2sign_pipeline.zip`. It contains one top level
folder named `how2sign_pipeline`. Extract it **outside** the ViPo-SLT repo.

### Linux or macOS server (including JupyterHub terminal)

```bash
# Pick a place that is not inside the repo, for example a tools folder.
mkdir -p ~/tools

# Extract; the archive creates ~/tools/how2sign_pipeline
unzip /path/to/how2sign_pipeline.zip -d ~/tools

# Move into the extracted folder.
cd ~/tools/how2sign_pipeline
```

### Windows PowerShell

```powershell
# Pick a place that is not inside the repo.
New-Item -ItemType Directory -Force -Path C:\work\tools | Out-Null

# Extract; the archive creates C:\work\tools\how2sign_pipeline
Expand-Archive -Path C:\path\to\how2sign_pipeline.zip -DestinationPath C:\work\tools

# Move into the extracted folder.
Set-Location C:\work\tools\how2sign_pipeline
```

Expected tree (the important files):

```
how2sign_pipeline/
  how2sign_pipeline/        the importable package
  tests/                    standard library test suite
  scripts/install_into_repo.py
  scripts/how2sign_eval_static.py
  scripts/model_smoke.py
  patches/viposlt_integration.patch
  pyproject.toml
  README.md
  AUDIT.md
```

---

## 5. Step 1: Set the environment variables

Adjust the three values to your machine, then set them for the session.

### Linux or macOS (bash, zsh)

```bash
export VIPOSLT_REPO="$HOME/serverViPo-SLT"
export HOW2SIGN_ROOT="$HOME/data/how2sign"
export VIPOSLT_FEATURE_ROOT="$HOME/data/how2sign_features"
```

### JupyterHub notebook cell

```python
%env VIPOSLT_REPO=/home/jovyan/serverViPo-SLT
%env HOW2SIGN_ROOT=/home/jovyan/data/how2sign
%env VIPOSLT_FEATURE_ROOT=/home/jovyan/data/how2sign_features
```

### Windows PowerShell

```powershell
$env:VIPOSLT_REPO = "C:\work\serverViPo-SLT"
$env:HOW2SIGN_ROOT = "C:\data\how2sign"
$env:VIPOSLT_FEATURE_ROOT = "C:\data\how2sign_features"
```

### Windows Command Prompt

```bat
REM Set the three variables for this terminal session.
set VIPOSLT_REPO=C:\work\serverViPo-SLT
set HOW2SIGN_ROOT=C:\data\how2sign
set VIPOSLT_FEATURE_ROOT=C:\data\how2sign_features
```

Running commands in a notebook: prefix shell commands with `!`, for example
`!python -m how2sign_pipeline inspect`. Environment variables set with `%env`
are visible to those `!` commands.

---

## 6. Step 2: Sanity check Python and git

These commands only check versions. They do not change anything.

```bash
python3 --version
git --version
```

On Windows use `python` if `python3` is not on PATH.

Expected output is like:

```
Python 3.11.2
git version 2.39.2
```

Also confirm the three variables point where you think they do.

```bash
echo "$VIPOSLT_REPO"
echo "$HOW2SIGN_ROOT"
echo "$VIPOSLT_FEATURE_ROOT"
```

The ViPo-SLT repo must be a git checkout, because the installer uses `git apply`.

```bash
git -C "$VIPOSLT_REPO" rev-parse --show-toplevel
```

Expected: it prints the absolute path to `$VIPOSLT_REPO`. If this fails, the repo
was copied without its `.git` folder. In that case either clone it properly
(`git clone <url>`), or apply the patch manually as described at the end of
Step 5.

Ensure the output folder is outside the repo. The command below prints nothing
when that is safe.

```bash
python3 -c "import os; r=os.path.realpath(os.environ['VIPOSLT_REPO']); o=os.path.realpath(os.environ['VIPOSLT_FEATURE_ROOT']); print('OVERLAP!' if o==r or r in o or o in r else 'paths are independent')"
```

Expected: `paths are independent`.

All commands in the following steps are run from the pipeline folder, unless a
step says otherwise.

```bash
cd ~/tools/how2sign_pipeline
```

---

## 7. Step 3: Inspect the dataset (read only)

This reads the How2Sign annotation tables and the media inventory. It writes
nothing to the dataset.

```bash
python3 -m how2sign_pipeline inspect --report "$VIPOSLT_FEATURE_ROOT/inspect_report.json"
```

What it looks for (case and whitespace tolerant, comma, tab, semicolon or pipe
delimited):

```
VIDEO_NAME  SENTENCE_NAME  START_REALIGNED  END_REALIGNED  SENTENCE
```

Optional columns used only for a fallback: `START`, `END`, `SIGNER_ID`.

Expected output, a JSON summary printed to the terminal:

```json
{
  "dataset_root": "/home/jovyan/data/how2sign",
  "samples": 35179,
  "errors": [],
  "splits": { "train": 31268, "val": 1715, "test": 2196 },
  "columns": { "train": ["VIDEO_NAME", "SENTENCE_NAME", "START_REALIGNED", "END_REALIGNED", "SENTENCE", "START", "END", "SIGNER_ID"] },
  "active_modalities": ["rgb_front: CLIP spatial", "rgb_front: VideoMAE motion"],
  "preserved_modalities": "All source files indexed; side/pose/depth are not consumed"
}
```

The exact counts depend on your copy of the dataset. The critical field is
`"errors": []`. If `errors` is not empty, each entry names the split or the
sample and the exact problem. Fix those before continuing. To make the command
fail loudly on errors, add `--strict`.

Useful extras:

- `--probe` measures each video with `ffprobe` and checks that every realigned
  crop fits inside the real video duration. Requires `ffprobe` on PATH.
- `--allow-sentence-clips` enables a fallback that derives the crop from a
  sentence level clip when the full source video is not present.
- `--splits train,val` limits the inspection to some splits.

---

## 8. Step 4: Prepare configs and manifests

This builds every file ViPo-SLT needs except the features themselves.

```bash
mkdir -p "$VIPOSLT_FEATURE_ROOT"
python3 -m how2sign_pipeline prepare
```

You can also pass the paths explicitly instead of relying on the variables.

```bash
python3 -m how2sign_pipeline prepare \
  --repo-root "$VIPOSLT_REPO" \
  --dataset-root "$HOW2SIGN_ROOT" \
  --output-root "$VIPOSLT_FEATURE_ROOT"
```

Expected output, a JSON summary:

```json
{
  "output_root": "/home/jovyan/data/how2sign_features",
  "fingerprint": "9c1f...ef42",
  "counts": { "train": 31268, "val": 1715, "test": 2196 },
  "modes": { "train": "train", "val": "dev", "test": "test" },
  "written": { "reused": 0, "written": 19 },
  "synthetic": false
}
```

Note the `modes` mapping: the official `val` split becomes the model's `dev`
split. `train` and `test` keep their names.

Files created inside `$VIPOSLT_FEATURE_ROOT`:

```
configs/How2Sign.yaml                          dataset info config
configs/how2sign_edm.yaml                      run config (dataset: How2Sign)
configs/pose_encoder.yaml                      minimal placeholder (pose is off)
source_manifest.json                           frozen inventory + fingerprint
annotations/How2Sign/train_info.npy            extractor manifest (+ .json sidecar)
annotations/How2Sign/val_info.npy
annotations/How2Sign/test_info.npy
preprocess/How2Sign/train_info_ml.npy          loader manifest (+ .json sidecar)
preprocess/How2Sign/dev_info_ml.npy
preprocess/How2Sign/test_info_ml.npy
preprocess/How2Sign/gloss_dict.npy             a legitimate empty {}
reports/prepare_report.json
```

Options:

- `--pilot` marks the run as a pilot (metadata only).
- `--force` overwrites existing artifacts (see Step 20). Re-running with the
  same dataset and no changes reuses identical files and reports
  `"written": 0`.

---

## 9. Step 5: Install the integration into the repo

This is the only step that touches the repository. Preview it first.

```bash
python3 scripts/install_into_repo.py \
  --repo-root "$VIPOSLT_REPO" \
  --output-root "$VIPOSLT_FEATURE_ROOT" \
  --apply-patch --link --dry-run
```

Expected output (nothing is changed yet):

```
APPLY patch /home/jovyan/tools/how2sign_pipeline/patches/viposlt_integration.patch
  diff --git a/ablation_visual.py b/ablation_visual.py
  @@ -36,11 +36,20 @@ def build_llm_args(args):
  diff --git a/dataset/loader_ft.py b/dataset/loader_ft.py
  ...
CREATE symlink (copy fallback) .../configs/.viposlt-....staging -> .../configs/How2Sign.yaml
RENAME stage .../configs/.viposlt-....staging -> .../configs/How2Sign.yaml
CREATE symlink (copy fallback) .../preprocess/.viposlt-....staging -> .../preprocess/How2Sign
RENAME stage .../preprocess/.viposlt-....staging -> .../preprocess/How2Sign
WRITE atomic marker .../\.viposlt_integration.json
Suggested .gitignore entries (not edited):
.viposlt_integration.json
configs/How2Sign.yaml
configs/how2sign_edm.yaml
preprocess/How2Sign
```

Now apply it for real.

```bash
python3 scripts/install_into_repo.py \
  --repo-root "$VIPOSLT_REPO" \
  --output-root "$VIPOSLT_FEATURE_ROOT" \
  --apply-patch --link
```

Expected: the same lines as the dry run, but the patch is now applied and the
three artifacts exist. What changed:

- Five repository files got the How2Sign support patch:
  `main.py`, `test.py`, `ablation_visual.py`, `dataset/loader_ft.py`,
  `models/edm_ddp.py`.
- `$VIPOSLT_REPO/configs/How2Sign.yaml` now points to the prepared dataset info
  config.
- `$VIPOSLT_REPO/configs/how2sign_edm.yaml` now points to the prepared run
  config used for training.
- `$VIPOSLT_REPO/preprocess/How2Sign/` now points to the prepared manifests.
- A marker file, `$VIPOSLT_REPO/.viposlt_integration.json`, records what was
  installed so `--revert` can undo it.

Confirm the patch is present:

```bash
git -C "$VIPOSLT_REPO" diff --name-only
```

Expected: exactly the five files listed above.

Notes:

- If any prepared artifact is missing, you will see
  `Missing prepared artifact ...; run prepare first`. Run Step 4 first.
- The installer prefers symlinks. Windows without symlink permission
  automatically falls back to copying. Windows junctions and reparse points are
  refused on purpose.
- It prints the `.gitignore` lines you may want to add. It never edits your
  `.gitignore` for you.
- To add the artifacts before you run `prepare`, apply only the patch:
  `python3 scripts/install_into_repo.py --apply-patch`. Then run Step 4, then run
  the installer again with `--link`.

Manual alternative if the repo has no `.git` folder:

```bash
cd "$VIPOSLT_REPO"
git init
git add -A
git commit -m "baseline before How2Sign integration"
git apply /home/jovyan/tools/how2sign_pipeline/patches/viposlt_integration.patch
```

Then copy the prepared configs and manifests into the repo. Run these from the
repository root so the relative destinations land in the right places.

```bash
cd "$VIPOSLT_REPO"
cp "$VIPOSLT_FEATURE_ROOT/configs/How2Sign.yaml" configs/How2Sign.yaml
cp "$VIPOSLT_FEATURE_ROOT/configs/how2sign_edm.yaml" configs/how2sign_edm.yaml
cp -r "$VIPOSLT_FEATURE_ROOT/preprocess/How2Sign" preprocess/How2Sign
```

On Windows, use `Copy-Item` in PowerShell instead of `cp`.

---

## 10. Step 6: Offline smoke test

This builds a tiny synthetic dataset, generates placeholder features, and runs
the **real** repository loader and collator against it. It uses standard library
mocks for numpy and torch, so it needs no GPU and no data.

```bash
python3 -m how2sign_pipeline smoke-test --repo-root "$VIPOSLT_REPO"
```

Expected output (trimmed):

```json
{
  "loader": {
    "backend": "explicit stdlib numpy/torch mocks",
    "batches": [
      { "split": "train", "ids": ["h...0", "h...1"], "batch_shapes": { "clip": [2, 3, 2048], "videomae": [2, 3, 1024] }, "targets_match": true, "padding_verified": true },
      { "split": "dev",   "targets_match": true, "padding_verified": true },
      { "split": "test",  "targets_match": true, "padding_verified": true }
    ]
  },
  "validation": { "ok": true, "errors": [] },
  "verification": "stdlib-fixture-tested"
}
```

What this proves: the loader discovers the manifest ids, returns six element
samples, collates finite tensors with the right last dimensions, uses the
official English sentence as the target, and pads variable length sequences
correctly. Add `--keep` to keep the temporary folder for inspection.

---

## 11. Step 7: Extract features on the GPU

This stage runs the repository's existing CLIP and VideoMAE extractors. It is
offline. Make sure the two models are already cached, for example:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-large-patch14'); snapshot_download('MCG-NJU/videomae-large')"
```

If the models live in a custom cache, pass it with `--cache-dir` or set
`HF_HUB_CACHE`.

First, a dry run that only prints the plan and touches nothing.

```bash
python3 -m how2sign_pipeline extract \
  --features clip,videomae --splits train --device cuda:0 --dry-run
```

Expected output:

```json
{
  "plan": {
    "features": ["clip", "videomae"],
    "splits": ["train"],
    "device": "cuda:0",
    "limit": null,
    "output": "/home/jovyan/data/how2sign_features"
  },
  "samples": 0
}
```

Next, a tiny real extraction to prove the whole path end to end. Two samples,
one feature at a time.

```bash
python3 -m how2sign_pipeline extract \
  --features clip --splits train --device cuda:0 --limit 2 --resume
```

Expected: a small JSON summary, then feature files under

```
$VIPOSLT_FEATURE_ROOT/features/clip-vit-large-patch14_feat_How2Sign/train/<id>.npy
$VIPOSLT_FEATURE_ROOT/features/mae_feat_How2Sign/train/<id>.npy
```

Each array is shape `[number_of_frames, 2048]` for CLIP and `[number_of_frames, 1024]`
for VideoMAE. The per sample report is written to
`$VIPOSLT_FEATURE_ROOT/extraction_report.json`, with one entry per id and its
shape and status.

Now run the full extraction. It resumes, so you can rerun it safely after an
interruption.

```bash
python3 -m how2sign_pipeline extract \
  --repo-root "$VIPOSLT_REPO" \
  --output-root "$VIPOSLT_FEATURE_ROOT" \
  --features clip,videomae --splits train,val,test \
  --device cuda:0 --batch-size 8 --resume
```

Expected: it processes the pending samples, skips already valid ones
(`"status": "reused"`), and writes `extraction_report.json`. It never downloads
anything and never fabricates a feature for a failed decode. A corrupt array is
renamed to `*.corrupt-<n>.npy` and re-extracted.

Notes:

- `--batch-size` bounds memory. It is not "one model per sample".
- `--limit N` caps the number of samples per split. Use it for pilots.
- `--device cpu` runs without a GPU (very slow) and is useful for a 1 sample
  check.
- Output `val` is written to the model's `dev` folder automatically.

---

## 12. Step 8: Validate the artifacts

This checks the loader contract across everything you produced.

```bash
python3 -m how2sign_pipeline validate
```

Expected output, ending in `"ok": true` and an empty `errors` list:

```json
{
  "output_root": "/home/jovyan/data/how2sign_features",
  "synthetic": false,
  "ok": true,
  "errors": [],
  "splits": {
    "train": { "source_split": "train", "count": 31268 },
    "dev":   { "source_split": "val",   "count": 1715 },
    "test":  { "source_split": "test",  "count": 2196 }
  },
  "fingerprint": "9c1f...ef42"
}
```

What it verifies: every manifest id maps to exactly one feature file and back,
no missing and no unexpected files, last dimensions equal the configured widths
(2048 and 1024), arrays are finite floating point, the English target is present
and equals `en_text`, and the gloss is empty.

Useful flags:

- `--no-features` checks only the configs and manifests, useful before
  extraction for a pilot.
- `--no-strict` prints the report but exits 0 even when there are errors. Use it
  to collect a full list of problems.

---

## 13. Step 9: Train

Run this from the repository root, because the loader reads `./preprocess/...`
relative to the current directory.

```bash
cd "$VIPOSLT_REPO"
python main.py --config configs/how2sign_edm.yaml
```

Expected: the usual ViPo-SLT training log. It loads `configs/How2Sign.yaml` (the
installed artifact), builds the dataset for `train` and `dev`, and starts
training. The run config sets `dataset: How2Sign`, `include_ctc: false`,
`include_pose: false`, English target, `sp_hidden_size: 2048`, and
`mo_hidden_size: 1024`.

---

## 14. Optional: run everything at once

You can chain inspect, prepare, extract, and strict validation in a single
command.

```bash
python3 -m how2sign_pipeline all \
  --repo-root "$VIPOSLT_REPO" \
  --dataset-root "$HOW2SIGN_ROOT" \
  --output-root "$VIPOSLT_FEATURE_ROOT" \
  --features clip,videomae --device cuda:0 --resume
```

Expected: a final JSON like:

```json
{
  "prepared": true,
  "validated": true,
  "splits": { "train": { "count": 31268 }, "dev": { "count": 1715 }, "test": { "count": 2196 } }
}
```

This does not run the installer. Apply the integration (Step 5) separately if you
have not already.

---

## 15. Optional: one real GPU model step

After real features exist, this runs exactly one forward, backward, and optimizer
step of the real model on one batch, with no downloads and no saved weights.

```bash
python3 scripts/model_smoke.py \
  --repo-root "$VIPOSLT_REPO" \
  --output-root "$VIPOSLT_FEATURE_ROOT" \
  --device cuda:0 --batch-size 2
```

Expected output:

```json
{"status": "PASS", "verification": "GPU-tested", "batch_ids": ["h...0", "h...1"], "loss": 12.34, "finite_gradients": true, "optimizer_steps": 1}
```

It requires cached LLM weights in the Hugging Face cache and the repo's cached
assets under `$VIPOSLT_REPO/data/models`. It refuses synthetic fixtures.

---

## 16. Optional: offline static gate

This is a fast, standard library only check of the repository integration and
the generated configs. It prints one line per check and exits non zero on any
failure.

```bash
python3 scripts/how2sign_eval_static.py \
  --repo-root "$VIPOSLT_REPO" \
  --output-root "$VIPOSLT_FEATURE_ROOT"
```

Expected output:

```
PASS compiled: external package and five integration modules
PASS static-checked: dataset and run config keys
PASS static-checked: manifest discovery and period-safe IDs
PASS static-checked: How2Sign train/dev/test selection
PASS static-checked: instruction prompt excludes target variables
```

Run it against a clean, unpatched repo and the discovery and split checks are
expected to FAIL, which proves the checks are real and not hardcoded.

---

## 17. Command reference

All pipeline commands:

| Command | What it does | Writes files |
| --- | --- | --- |
| `inspect` | Reads annotations and media. Reports errors. | only with `--report` |
| `prepare` | Builds configs, manifests, empty gloss dict. | yes, under `--output-root` |
| `smoke-test` | Tiny synthetic run through the real loader. | temporary (unless `--keep`) |
| `extract` | Runs CLIP and VideoMAE on the GPU. Resumable, offline. | feature arrays and reports |
| `validate` | Verifies the loader contract. | a validation report |
| `all` | inspect, prepare, extract, validate. | yes |

Common flags:

| Flag | Meaning |
| --- | --- |
| `--repo-root` | ViPo-SLT checkout. Defaults to `VIPOSLT_REPO`. |
| `--output-root` | Output folder. Defaults to `VIPOSLT_FEATURE_ROOT`. |
| `--dataset-root` | How2Sign root. Defaults to `HOW2SIGN_ROOT`. |
| `--splits` | Comma list `train,val,test`. `dev` is accepted as an alias for `val`. |
| `--features` | Comma list `clip,videomae`. |
| `--device` | `cuda:0` or `cpu`. |
| `--batch-size` | Extraction batch size, bounds memory. |
| `--limit` | Max samples per split, for pilots. |
| `--cache-dir` | Hugging Face cache directory. |
| `--resume` | Reuse valid existing features. |
| `--force` | Overwrite or quarantine conflicting output. |
| `--dry-run` | Print the plan, change nothing. |
| `--strict` | Inspect only: fail on errors instead of reporting them. Validate is already strict by default. |

Installer flags:

| Flag | Meaning |
| --- | --- |
| `--apply-patch` | Apply the integration patch to the repo. |
| `--link` | Create the repo relative artifacts (configs and preprocess). |
| `--revert` | Reverse the patch and remove managed artifacts. |
| `--force` | Allow applying on a dirty tree, or back up a pre-existing artifact. |
| `--dry-run` | Print exactly what would change. |

Get help for anything:

```bash
python3 -m how2sign_pipeline --help
python3 -m how2sign_pipeline extract --help
python3 scripts/install_into_repo.py --help
```

---

## 18. Understanding the output layout

Everything under `$VIPOSLT_FEATURE_ROOT`:

| Path | Role | Safe to regenerate |
| --- | --- | --- |
| `source_manifest.json` | Frozen inventory and fingerprint. | yes, with `prepare` |
| `configs/How2Sign.yaml` | Dataset info config read by `main.py`. Path only. | yes |
| `configs/how2sign_edm.yaml` | Run config for training. | yes |
| `configs/pose_encoder.yaml` | Placeholder, never loaded while pose is off. | yes |
| `annotations/How2Sign/<split>_info.npy` | Extractor manifests with `folder` and `original_info`. | yes |
| `preprocess/How2Sign/{train,dev,test}_info_ml.npy` | Loader manifests. | yes |
| `preprocess/How2Sign/gloss_dict.npy` | Genuine empty `{}` gloss dictionary. | yes |
| `features/<feature_dir>/<mode>/<id>.npy` | The feature arrays used by training. | per feature, resumable |
| `extraction_jobs/<fingerprint>/How2Sign/` | Per run extractor manifests. | yes |
| `features/<feature_dir>/extraction.json` | Cache compatibility fingerprint. | yes |
| `extraction_report.json` | Per sample status and shapes. | yes |
| `reports/prepare_report.json` | Summary of the prepare step. | yes |
| `reports/validation_report.json` | Summary of the validate step. | yes |
| `integration-backups/<id>/` | Backups of pre-existing in-repo artifacts. | no, this is user data |
| `.extraction.lock` | Prevents two extractions in one output folder. | yes |

The fingerprint in `source_manifest.json` lets the tools detect that the source
data changed, without hashing the whole dataset every time.

---

## 19. Modalities

- **Used by the default model:** frontal RGB, as CLIP spatial features
  (`sp_hidden_size: 2048`) and VideoMAE motion features (`mo_hidden_size: 1024`).
- **Preserved but not used:** side camera, pose, depth. Pose mode is not
  supported in the baseline and is set to `include_pose: false`.
- **Glosses and CTC:** `include_ctc` is `false`. How2Sign has no verified
  glosses, so the gloss dictionary is a genuine empty `{}`. The pipeline never
  invents gloss labels.

---

## 20. Resume, force, and safety

- **Prepare is resumable.** Re-running with unchanged source reuses identical
  files and reports `"written": 0`. If a file exists with different content, the
  run stops and asks for `--force`.
- **Extraction is resumable.** Valid existing arrays are reused. Corrupt arrays
  are quarantined and re-extracted. If the model, layer, temporal sampling, or
  crop settings change, the feature cache fingerprint no longer matches and the
  old feature folder is quarantined (with `--force`) instead of being mixed.
- **No downloads.** Extraction sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`.
- **No deletion of raw data.** The dataset is only ever read.
- **Extraction lock.** A second extraction against the same output folder fails
  fast with `Another extraction owns ...`.

---

## 21. Troubleshooting

| Message | Meaning and fix |
| --- | --- |
| `--repo-root does not look like the ViPo-SLT checkout` | The path has no `main.py` or `dataset/loader_ft.py`. Point `VIPOSLT_REPO` at the repo root. |
| `output-root must be outside the repository` | Move `VIPOSLT_FEATURE_ROOT` so it is not inside, above, or equal to the repo. |
| `Repository is dirty` | The repo has uncommitted changes and a patch must be applied. Commit or stash them, or pass `--force` if you are sure. |
| `No integration marker; refusing an unmanaged revert` | There is nothing to revert. Revert only works after an install created the marker. |
| `Patch contains unexpected repository files` | A custom patch touches files outside the allowlist. Use the shipped patch, or update the allowlist intentionally. |
| `Missing prepared artifact ...; run prepare first` | `--link` needs the configs and manifests. Run Step 4 first. |
| `Real extraction requires repository dependencies ...` | The repo environment is missing numpy, torch, transformers, av, or Pillow. Activate the repo environment. |
| `CUDA requested but unavailable` | No GPU is visible. Check `nvidia-smi`, or use `--device cpu` for a tiny test. |
| `Model assets must already be cached` | Download the CLIP and VideoMAE models into the Hugging Face cache, or pass `--cache-dir`. |
| `missing N feature file(s), e.g. [...]` | Extraction is incomplete for those ids. Re-run `extract --resume`. |
| `last dim X != configured Y` | A feature width does not match the run config. Fix the extractor settings, or update the measured width in `configs/how2sign_edm.yaml` and document why. |
| `no samples for required split` | The requested split has no rows in the annotation table. Check the table name and `--splits`. |
| `another extraction owns ...` | Another run holds the lock. Wait for it to finish, or reuse its output folder. |
| Symlink permission denied on Windows | Expected. The installer falls back to copying automatically. |
| `Dataset root does not exist` | `HOW2SIGN_ROOT` or `--dataset-root` is wrong. |
| `duplicate sample ID` | The annotation tables repeat a `SENTENCE_NAME`. Clean the source tables. |

Run any command with `--help` for the full flag list and a one line description.

---

## 22. Running the test suite

The suite is standard library only and mocks numpy and torch, so it runs without
a GPU and without the dataset. Tests that need a real checkout are skipped unless
`VIPOSLT_REPO` points at one.

```bash
VIPOSLT_REPO="$VIPOSLT_REPO" python3 -m unittest discover -s tests -v
```

Expected, on Linux or macOS:

```
Ran 73 tests in 2.9s
OK (skipped=1)
```

The single skip is the numpy only resume test. On Windows use `python` instead of
`python3`.

You can also run the offline static gate from Step 16 as part of your checks.

---

## 23. Uninstalling and reverting

To undo the repository integration, reverse the patch and remove the managed
artifacts. Raw data and your output folder are left untouched.

```bash
python3 scripts/install_into_repo.py --repo-root "$VIPOSLT_REPO" --revert --dry-run
python3 scripts/install_into_repo.py --repo-root "$VIPOSLT_REPO" --revert
```

Expected output:

```
RUN git -C /home/jovyan/serverViPo-SLT apply --reverse /home/jovyan/tools/how2sign_pipeline/patches/viposlt_integration.patch
REMOVE managed artifact /home/jovyan/serverViPo-SLT/preprocess/How2Sign
REMOVE managed artifact /home/jovyan/serverViPo-SLT/configs/how2sign_edm.yaml
REMOVE managed artifact /home/jovyan/serverViPo-SLT/configs/How2Sign.yaml
REMOVE marker /home/jovyan/serverViPo-SLT/.viposlt_integration.json
```

One important detail: if the repo **already** contained the integration patch
before you ran the installer (for example, if you applied it by hand), the
installer treats it as pre-existing and does not claim ownership of it. In that
case `--revert` removes the artifacts and the marker but deliberately leaves your
patch in place, so your own work is not discarded.

After reverting, confirm the tree is clean:

```bash
git -C "$VIPOSLT_REPO" status --porcelain
```

Expected: no output.

---

## 24. Known limitations

- No gloss labels, so CTC is disabled by default.
- Pose mode is not supported and fails fast if requested.
- Only the frontal camera is used. Other views are preserved but unused.
- Expected feature widths are validated against the run config (2048 and 1024).
  If a different extractor setting yields another width, update the run config
  to the measured value and document it.
- Samples whose frontal video is missing or ambiguous are reported, never
  guessed.
- The integration must be applied with `git`. A repo without a `.git` folder
  needs the manual steps in Step 5.

---

## 25. Licence and provenance

How2Sign and ViPo-SLT keep their original licences and attribution. This
pipeline only reads the corpus and generates derived feature caches. It does not
relicense the data. Preserve the upstream licence notices when you redistribute
any generated artifacts.
