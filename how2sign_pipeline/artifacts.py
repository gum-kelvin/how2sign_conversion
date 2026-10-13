"""Prepare and validate the generated How2Sign artifacts.

Everything here is deterministic and standard-library only so it can be audited
without numpy/torch. Object manifests are written as real NPY files by
``npyio.write_object``; a parallel JSON sidecar (numeric keys stringified) makes
the same content machine-readable offline and is what the mocked loader reads.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

from .npyio import atomic_bytes, read_float, write_object

SPLIT_TO_MODE = {'train': 'train', 'val': 'dev', 'test': 'test'}
SOURCE_SPLITS = ('train', 'val', 'test')
MODES = ('train', 'dev', 'test')

REQUIRED_INFO_KEYS = ('dataset_root', 'dict_path', 'feature_dir', 'sp_path', 'mo_path',
                      'phm_path', 'pose_path', 'pose_encoder_config_path', 'language')

FEATURES = (('clip', 'clip-vit-large-patch14_feat_How2Sign', 'sp_hidden_size', 'include_sp'),
            ('videomae', 'mae_feat_How2Sign', 'mo_hidden_size', 'include_mo'))

# Derived from configs/baseline_edm.yaml, made path-portable and gloss/pose-free.
DEFAULT_RUN_CONFIG = {
    'random_seed': 0, 'mode': 'train', 'batch_size': 8, 'accumulate_grad_batches': 4,
    'test_batch_size': 2, 'num_worker': 0, 'device': 'cuda:0',
    'feeder': 'dataset.loader_img.BaseFeeder',
    'feeder_ft': 'dataset.loader_ft.PrecomputedFeatureDataset',
    'dataset': 'How2Sign', 'include_sp': True, 'include_pose': False, 'include_mo': True,
    'sp_hidden_size': 2048, 'pose_hidden_size': 512, 'mo_hidden_size': 1024,
    'phase': 'train', 'num_epoch': 100, 'num_warmup_epochs': 10, 'epochs_before_lr_decay': 10,
    'scheduler_patience': 5, 'validation_interval': 3, 'patience': 10, 'monitor_bleu': True,
    'score_dependent_scheduler': True, 'contrastive': True, 'include_ctc': False,
    'llm_precision': 'fp32', 'gradient_checkpointing': False, 'ctc_weight': 1.0,
    'language': 'English',
    'optimizer_args': {'optimizer': 'Adam', 'base_lr': 0.0002, 'learning_ratio': 1,
                       'gamma': 0.2, 'weight_decay': 0.0001, 'start_epoch': 0},
    'feeder_args': {'mode': 'train', 'datatype': 'video', 'num_gloss': -1, 'drop_ratio': 1.0,
                    'frame_interval': 1, 'image_scale': 1.0, 'input_size': 224},
    'slr': 'modules.slr.SLRModel', 'slr_weights': '',
    'model': 'models.edm_ddp.EDM', 'llm': 'bigscience/mt0-large',
    'temporal_args': {'num_layers': 4, 'num_heads': 4},
    'slr_args': {'c2d_type': 'swinb_mstcn-3', 'conv_type': 2, 'use_bn': 1,
                 'share_classifier': 1, 'weight_norm': True},
    'llm_args': {'apply_lora': True, 'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': 0.1,
                 'llm_hidden_size': 768, 'max_txt_len': 128, 'tuning_mode': 'lora',
                 'prompt': 'Translate the given sentence into'},
}

POSE_ENCODER_CONFIG = {'model': {'RecognitionNetwork': {'enabled': False,
                         'note': 'placeholder; How2Sign pose mode is not supported'}}}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + '\n').encode('utf-8')


def _sidecar_manifest(manifest):
    """Stringify numeric manifest keys so JSON keeps them (loader tolerates either)."""
    return {('prefix' if key == 'prefix' else str(key)): value for key, value in manifest.items()}


def _stable_source(source):
    """The part of a source manifest that defines its identity (no timestamps)."""
    return {'dataset_root': source.get('dataset_root'), 'splits': list(source.get('splits') or []),
            'samples': source.get('samples') or [], 'synthetic': bool(source.get('synthetic')),
            'pilot': bool(source.get('pilot'))}


def _fingerprint(source):
    """Deterministic fingerprint over the timestamp-free source identity."""
    return hashlib.sha256(_json_bytes(_stable_source(source))).hexdigest()


def _write(path, data, force, resume=True):
    """Atomic, deterministic write; identical content is a no-op when resuming."""
    path = Path(path)
    if path.is_file() and path.read_bytes() == data:
        if resume:
            return 'reused'
        atomic_bytes(path, data)
        return 'written'
    if path.exists() and not force:
        raise FileExistsError(f'{path} already exists with different content; use --force to overwrite')
    atomic_bytes(path, data)
    return 'written'


def _write_json(path, data, force, resume=True):
    return _write(path, _json_bytes(data), force, resume)


def _write_object(path, manifest, force, resume=True):
    """Write an object NPY deterministically so resume can detect identical content."""
    from .npyio import object_bytes
    return _write(path, object_bytes(manifest), force, resume)


def _check_no_overlap(out, raw_root, repo_root):
    out = Path(out).resolve()
    if repo_root is not None:
        repo = Path(repo_root).resolve()
        if out == repo or repo in out.parents or out in repo.parents:
            raise ValueError('output-root must be outside the repository')
    if raw_root is not None:
        raw = Path(raw_root).resolve()
        if out == raw or raw in out.parents or out in raw.parents:
            raise ValueError('output-root must not overlap the raw dataset')


def _load_report(report):
    if isinstance(report, (str, os.PathLike)):
        report = json.loads(Path(report).read_text(encoding='utf-8-sig'))
    if not isinstance(report, dict):
        raise ValueError('report must be a mapping or a path to a JSON report')
    return report


def _record_fields(sample, source_split, mode):
    return {
        'fileid': sample['fileid'], 'source_id': sample['source_id'],
        'source_video': sample['source_video'], 'source_split': source_split, 'split': mode,
        'folder': sample['folder'], 'text': sample['text'], 'en_text': sample['en_text'],
        'gloss': sample.get('gloss', ''), 'start': sample['start'], 'end': sample['end'],
        'signer': sample.get('signer'), 'timing_reference': sample.get('timing_reference', 'source_video'),
        'original_info': sample['original_info'],
    }


def prepare(report, output_root, *, repo_root=None, force=False, resume=True, synthetic=False,
            pilot=False, run_overrides=None, splits=SOURCE_SPLITS):
    """Build configs, extractor manifests, loader manifests and the gloss dictionary.

    Returns a summary dict. Shared artifacts are written with stable IDs so source
    ``val`` becomes model ``dev`` while the source split name is preserved.
    """
    report = _load_report(report)
    errors = list(report.get('errors') or [])
    if errors:
        raise ValueError(f'inspection reported {len(errors)} error(s); refusing to prepare: {errors[:5]}')
    samples = list(report.get('samples') or [])
    if not samples:
        raise ValueError('inspection report contains no usable samples')
    raw_root = report.get('dataset_root')
    out = Path(os.path.expandvars(str(output_root))).expanduser().resolve()
    _check_no_overlap(out, raw_root, repo_root)
    force = bool(force)
    resume = bool(resume)

    requested = [s for s in splits if s in SOURCE_SPLITS]
    if set(requested) != set(splits):
        raise ValueError(f'splits must be a subset of {SOURCE_SPLITS}')
    for split in requested:
        if not any(s['source_split'] == split for s in samples):
            raise ValueError(f'no samples for required split {split}')

    source_manifest = {'synthetic': bool(synthetic), 'pilot': bool(pilot),
                       'dataset_root': str(Path(raw_root).resolve()) if raw_root else None,
                       'splits': list(requested), 'samples': samples}
    fingerprint = _fingerprint(source_manifest)
    source_manifest['fingerprint'] = fingerprint
    counts = {split: sum(1 for s in samples if s['source_split'] == split) for split in requested}

    written = {'reused': 0, 'written': 0}
    def record(status):
        written[status] += 1

    record(_write_json(out / 'source_manifest.json', source_manifest, force, resume))

    produced = {'annotations': {}, 'loader_manifests': {}, 'modes': {}}
    for split in requested:
        mode = SPLIT_TO_MODE[split]
        produced['modes'][split] = mode
        recs = [s for s in samples if s['source_split'] == split]
        extractor = {'prefix': _record_fields(recs[0], split, mode).copy()}
        for special in ('folder', 'text', 'en_text', 'gloss'):
            extractor['prefix'].pop(special, None)
        loader = {'prefix': ''}
        for index, sample in enumerate(recs):
            extractor[index] = _record_fields(sample, split, mode)
            loader[index] = {'fileid': sample['fileid'], 'text': sample['text'],
                             'en_text': sample['en_text'], 'gloss': sample.get('gloss', '')}
        anno_path = out / 'annotations' / 'How2Sign' / f'{split}_info.npy'
        record(_write_object(anno_path, extractor, force, resume))
        record(_write_json(anno_path.with_suffix('.json'), _sidecar_manifest(extractor), force, resume))
        load_path = out / 'preprocess' / 'How2Sign' / f'{mode}_info_ml.npy'
        record(_write_object(load_path, loader, force, resume))
        record(_write_json(load_path.with_suffix('.json'), _sidecar_manifest(loader), force, resume))
        produced['annotations'][split] = str(anno_path)
        produced['loader_manifests'][mode] = str(load_path)

    gloss_path = out / 'preprocess' / 'How2Sign' / 'gloss_dict.npy'
    record(_write_object(gloss_path, {}, force, resume))
    record(_write_json(gloss_path.with_suffix('.json'), {}, force, resume))

    pose_cfg = out / 'configs' / 'pose_encoder.yaml'
    record(_write_json(pose_cfg, POSE_ENCODER_CONFIG, force, resume))

    features_root = out / 'features'
    info = {
        'dataset': 'How2Sign', 'language': 'English',
        'dataset_root': str(Path(raw_root).resolve()) if raw_root else '',
        'dict_path': str(gloss_path),
        'feature_dir': str(features_root),
        'sp_path': str(features_root / 'clip-vit-large-patch14_feat_How2Sign'),
        'mo_path': str(features_root / 'mae_feat_How2Sign'),
        'phm_path': str(features_root / 'mae_feat_How2Sign'),
        'pose_path': str(out / 'pose'),
        'pose_encoder_config_path': str(pose_cfg),
    }
    run = dict(DEFAULT_RUN_CONFIG)
    run['dataset'] = 'How2Sign'
    run['work_dir'] = str(out / 'work_dir')
    if run_overrides:
        run.update(run_overrides)
    if run.get('include_ctc'):
        raise ValueError('How2Sign has no verified glosses; keep include_ctc: false')
    if run.get('include_pose'):
        raise ValueError('How2Sign pose conversion is not supported; keep include_pose: false')
    record(_write_json(out / 'configs' / 'How2Sign.yaml', info, force, resume))
    record(_write_json(out / 'configs' / 'how2sign_edm.yaml', run, force, resume))

    prepare_report = {'created': _now(), 'synthetic': bool(synthetic), 'pilot': bool(pilot),
                      'fingerprint': fingerprint, 'counts': counts, 'modes': produced['modes'],
                      'dataset_root': source_manifest['dataset_root'],
                      'features': [name for name, _, _, enabled in FEATURES if run.get(enabled)],
                      'widths': {name: run[width_key] for name, _, width_key, _ in FEATURES},
                      'written': written}
    _write_json(out / 'reports' / 'prepare_report.json', prepare_report, True, resume)
    return {'output_root': str(out), 'fingerprint': fingerprint, 'counts': counts,
            'modes': produced['modes'], 'manifest': produced, 'written': written,
            'synthetic': bool(synthetic)}


def _read_sidecar(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _digits(mapping):
    return [value for key, value in mapping.items() if str(key).isdigit()]


def validate(output_root, *, repo_root=None, strict=True, require_features=True):
    """Verify the generated artifacts beyond mere existence.

    Returns a report dict and, when ``strict`` and problems exist, raises so the
    CLI exits non-zero.
    """
    out = Path(os.path.expandvars(str(output_root))).expanduser().resolve()
    errors, notes = [], []
    source_path = out / 'source_manifest.json'
    if not source_path.is_file():
        raise FileNotFoundError(f'missing {source_path}; run prepare first')
    source = _read_sidecar(source_path)
    samples = source.get('samples') or []
    synthetic = bool(source.get('synthetic'))
    _check_no_overlap(out, source.get('dataset_root'), repo_root)

    info = _read_sidecar(out / 'configs' / 'How2Sign.yaml')
    run = _read_sidecar(out / 'configs' / 'how2sign_edm.yaml')
    missing = [key for key in REQUIRED_INFO_KEYS if key not in info]
    if missing:
        errors.append(f'How2Sign.yaml missing keys: {missing}')
    if run.get('dataset') != 'How2Sign':
        errors.append("run config dataset must be exactly 'How2Sign'")
    if run.get('include_ctc'):
        errors.append('run config must set include_ctc: false (no verified glosses)')
    if run.get('include_pose'):
        errors.append('run config must set include_pose: false (pose unsupported)')

    split_report = {}
    for split in SOURCE_SPLITS:
        source_ids = [s['fileid'] for s in samples if s['source_split'] == split]
        if not source_ids:
            continue
        mode = SPLIT_TO_MODE[split]
        load_side = out / 'preprocess' / 'How2Sign' / f'{mode}_info_ml.json'
        if not load_side.is_file():
            errors.append(f'missing loader manifest sidecar {load_side}')
            continue
        manifest = _read_sidecar(load_side)
        header = 'prefix' in manifest
        records = _digits(manifest)
        target_ids = [r['fileid'] for r in records]
        if not header:
            errors.append(f'{mode}_info_ml manifest has no prefix header')
        if len(target_ids) != len(set(target_ids)):
            errors.append(f'{mode}_info_ml has duplicate fileids')
        if set(target_ids) != set(source_ids):
            errors.append(f'{mode} id bijection mismatch: missing={sorted(set(source_ids) - set(target_ids))[:5]} '
                          f'unexpected={sorted(set(target_ids) - set(source_ids))[:5]}')
        keys = [str(k) for k in manifest if str(k).isdigit()]
        if sorted(int(k) for k in keys) != list(range(len(keys))):
            errors.append(f'{mode}_info_ml keys are not dense integers')
        for record in records:
            if not record.get('fileid'):
                errors.append(f'{mode}: record without fileid')
            text, en_text = record.get('text', ''), record.get('en_text', '')
            if not text or text != en_text:
                errors.append(f"{record.get('fileid')}: text must be a nonempty English target equal to en_text")
            if record.get('gloss', '') != '':
                errors.append(f"{record.get('fileid')}: gloss must be empty for gloss-free How2Sign")
        split_report[mode] = {'source_split': split, 'count': len(records)}

        anno_side = out / 'annotations' / 'How2Sign' / f'{split}_info.json'
        if not anno_side.is_file():
            errors.append(f'missing extractor manifest sidecar {anno_side}')
            continue
        extractor = _read_sidecar(anno_side)
        if 'prefix' not in extractor:
            errors.append(f'{split}_info extractor manifest has no prefix header')
        ext_records = _digits(extractor)
        for record in ext_records:
            if not record.get('folder'):
                errors.append(f"{record.get('fileid')}: extractor record missing folder")
            original = record.get('original_info') or {}
            for key in ('START_REALIGNED', 'END_REALIGNED'):
                value = original.get(key)
                if not isinstance(value, (int, float)) or value != value:
                    errors.append(f"{record.get('fileid')}: original_info.{key} is not a finite number")

    gloss = out / 'preprocess' / 'How2Sign' / 'gloss_dict.npy'
    if not gloss.is_file():
        errors.append(f'missing empty gloss dictionary {gloss}')
    else:
        gloss_side = gloss.with_suffix('.json')
        if gloss_side.is_file() and _read_sidecar(gloss_side) != {}:
            errors.append('gloss_dict sidecar is not an empty dictionary')

    if require_features and not synthetic:
        for name, dirname, width_key, enabled in FEATURES:
            if not run.get(enabled):
                continue
            width = run.get(width_key)
            expected = {}
            for split in SOURCE_SPLITS:
                ids = [s['fileid'] for s in samples if s['source_split'] == split]
                if ids:
                    expected[SPLIT_TO_MODE[split]] = ids
            for mode, ids in expected.items():
                target = out / 'features' / dirname / mode
                present = {p.stem for p in target.glob('*.npy')} if target.is_dir() else set()
                missing_ids = [i for i in ids if i not in present]
                extra = sorted(present - set(ids))
                if missing_ids:
                    errors.append(f'{name}/{mode}: missing {len(missing_ids)} feature file(s), e.g. {missing_ids[:5]}')
                if extra:
                    errors.append(f'{name}/{mode}: unexpected feature files {extra[:5]}')
                for fileid in ids:
                    path = target / f'{fileid}.npy'
                    if not path.is_file():
                        continue
                    try:
                        info_array = read_float(path)
                    except (ValueError, OSError, EOFError) as exc:
                        errors.append(f'{name}/{mode}/{fileid}: {exc}')
                        continue
                    rows, cols = info_array['shape']
                    if cols != width:
                        errors.append(f'{name}/{mode}/{fileid}: last dim {cols} != configured {width}')
    elif synthetic:
        notes.append('synthetic fixture: real feature arrays are not required for validation')

    prepare_path = out / 'reports' / 'prepare_report.json'
    prepare_report = _read_sidecar(prepare_path) if prepare_path.is_file() else {}
    source_fingerprint = _fingerprint(source)
    if source.get('fingerprint') not in (None, source_fingerprint):
        errors.append('source_manifest.json is internally inconsistent; regenerate artifacts')
    if prepare_report.get('fingerprint') and prepare_report['fingerprint'] != source_fingerprint:
        errors.append('source_manifest.json changed since prepare; regenerate artifacts')
    if prepare_report.get('fingerprint'):
        expected = prepare_report.get('counts') or {}
        actual = {s: sum(1 for x in samples if x['source_split'] == s) for s in SOURCE_SPLITS}
        for split, count in expected.items():
            if actual.get(split) != count:
                errors.append(f'source split {split} count changed: {actual.get(split)} != {count}')

    if not synthetic:
        for sample in samples:
            path = Path(sample.get('folder', ''))
            stat = sample.get('source_stat')
            if stat and path.is_file():
                if path.stat().st_size != stat.get('size'):
                    errors.append(f"{sample['fileid']}: source video size changed; regenerate")

    report = {'output_root': str(out), 'synthetic': synthetic, 'ok': not errors,
              'errors': errors, 'notes': notes, 'splits': split_report,
              'fingerprint': prepare_report.get('fingerprint')}
    _write_json(out / 'reports' / 'validation_report.json', report, True)
    if strict and errors:
        raise ValueError(f'validation failed with {len(errors)} error(s): {errors[:5]}')
    return report
