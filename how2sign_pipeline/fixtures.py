"""Generate a tiny, explicitly synthetic How2Sign output for offline smoke tests.

Nothing here represents real extracted features: the numeric arrays are obvious
placeholders written by ``npyio.write_float`` and the manifest carries
``synthetic: true`` so the real extractor refuses to consume them.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from . import artifacts
from .metadata import inspect_dataset
from .npyio import write_float


def _column_names():
    return ['VIDEO_NAME', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE',
            'START', 'END', 'SIGNER_ID']


def build_synthetic_source(root, *, splits=('train', 'val', 'test'), per_split=3):
    """Create a minimal official-shaped dataset (one frontal video per sentence)."""
    root = Path(root)
    for split in splits:
        table = root / f'how2sign_realigned_{split}.tsv'
        table.parent.mkdir(parents=True, exist_ok=True)
        with table.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=_column_names(), delimiter='\t')
            writer.writeheader()
            for index in range(per_split):
                video_name = f'source.{split}.{index}'
                video = root / split / 'rgb_front' / f'{video_name}-rgb_front.mp4'
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b'fixture placeholder; not a decodable video')
                writer.writerow({
                    'VIDEO_NAME': video_name,
                    'SENTENCE_NAME': f'{split}.sentence.{index}',
                    'START_REALIGNED': str(1.5 + index),
                    'END_REALIGNED': str(3.5 + index),
                    'SENTENCE': f'Official English, punctuation | café 雨 {split} {index}!',
                    'START': '0', 'END': '30', 'SIGNER_ID': f'person{index}',
                })
    return root


def build_synthetic_output(output_root, *, repo_root=None, splits=('train', 'val', 'test'),
                           per_split=3, force=True):
    """inspect -> prepare -> synthetic numeric features, all under/next to output_root."""
    out = Path(os.path.expandvars(str(output_root))).expanduser().resolve()
    # Keep the source tree beside (not inside) the output so overlap guards hold.
    source_root = out.parent / (out.name + '_synthetic_source')
    out.mkdir(parents=True, exist_ok=True)
    build_synthetic_source(source_root, splits=splits, per_split=per_split)
    report = inspect_dataset(source_root, splits=list(splits))
    if report['errors']:
        raise ValueError(f'synthetic source inspection failed: {report["errors"][:3]}')
    summary = artifacts.prepare(report, out, repo_root=repo_root, force=force, synthetic=True)

    import json
    run = json.loads((out / 'configs' / 'how2sign_edm.yaml').read_text(encoding='utf-8'))
    info = json.loads((out / 'configs' / 'How2Sign.yaml').read_text(encoding='utf-8'))

    for feature, dirname, width_key, enabled in artifacts.FEATURES:
        if not run.get(enabled):
            continue
        base = Path(info['sp_path'] if feature == 'clip' else info['mo_path'])
        for split in splits:
            mode = artifacts.SPLIT_TO_MODE[split]
            ids = [s['fileid'] for s in report['samples'] if s['source_split'] == split]
            for offset, fileid in enumerate(ids):
                target = base / mode / f'{fileid}.npy'
                target.parent.mkdir(parents=True, exist_ok=True)
                write_float(target, rows=2 + (offset % 3), width=run[width_key])
    summary['synthetic_source'] = str(source_root)
    return summary
