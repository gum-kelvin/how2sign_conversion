"""Bounded orchestration of the repository's existing CLIP and VideoMAE readers."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import sys
import time
from types import SimpleNamespace

FEATURES = {'clip': ('vit_extract_feature', 'clip-vit-large-patch14_feat_How2Sign', 'openai/clip-vit-large-patch14', 2048), 'videomae': ('mae_extract_feature', 'mae_feat_How2Sign', 'MCG-NJU/videomae-large', 1024)}


def check_array(array, width, sample):
    import numpy as np
    if array.ndim != 2 or not array.shape[0] or array.shape[-1] != width:
        raise ValueError(f'{sample}: expected nonempty [T,{width}], got {array.shape}')
    if array.dtype.kind != 'f' or not np.isfinite(array).all():
        raise ValueError(f'{sample}: features must be finite floating point values')
    return list(array.shape)


def _json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _extract(args):
    repo = Path(args.repo_root).resolve()
    out = Path(args.output_root).resolve()
    if out == repo or repo in out.parents or out in repo.parents:
        raise ValueError('output-root must be outside the repository')
    names = args.features.split(',') if isinstance(args.features, str) else args.features
    unknown = set(names) - set(FEATURES)
    if unknown or not names or len(names) != len(set(names)):
        raise ValueError(f'Unsupported features {sorted(unknown)}; supported: clip,videomae')
    splits = args.splits.split(',') if isinstance(args.splits, str) else args.splits
    splits = ['val' if s == 'dev' else s for s in splits]
    if not splits or len(splits) != len(set(splits)) or set(splits) - {'train', 'val', 'test'}:
        raise ValueError('splits must be train,val,test (dev aliases val)')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('batch-size and limit must be positive')
    plan = {'features': names, 'splits': splits, 'device': args.device, 'limit': args.limit, 'output': str(out)}
    if args.dry_run:
        return {'dry_run': True, **plan}
    # Ordinary extraction never initiates a model download.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    cache = getattr(args, 'cache_dir', None)
    if cache:
        os.environ['HF_HUB_CACHE'] = str(Path(cache).resolve())
    try:
        import numpy as np
        import torch
        import av
    except ImportError as exc:
        raise RuntimeError('Real extraction requires repository dependencies including numpy, torch, transformers, av and Pillow. No download was attempted.') from exc
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; select --device cpu or configure a CUDA server')
    source_path = out / 'source_manifest.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if source.get('synthetic') or any(s.get('synthetic') for s in source['samples']):
        raise ValueError('Synthetic fixture cannot be used for real extraction')
    raw = Path(source['dataset_root']).resolve()
    if out == raw or raw in out.parents or out in raw.parents:
        raise ValueError('output-root must be separate from raw data')
    assert_source_unchanged(source['samples'])
    sys.path.insert(0, str(repo))
    helpers = importlib.import_module('preprocess.SP_FT.helpers')
    helpers.av = av  # Current repository decoder omits this import.
    reports = []
    for feature in names:
        script, dirname, model, width = FEATURES[feature]
        module = importlib.import_module('preprocess.SP_FT.' + script)
        original_read = module.read_video
        def checked_read(fname, start_time=None, end_time=None):
            frames = original_read(fname, start_time=start_time, end_time=end_time)
            if not frames:
                raise ValueError(f'Video decode yielded zero frames: {fname}, crop={start_time}:{end_time}')
            return frames
        # Decoder guard is installed only during each iterator's lifetime.
        config = {'model': model, 'layer': -1, 'scales': [1, 2] if feature == 'clip' else [], 'overlap': 8, 'batch_size': args.batch_size, 'versions': dependency_versions(),
                  'source_hash': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  'code': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'preprocess/SP_FT'/f'{script}.py', repo/'preprocess/SP_FT/helpers.py', repo/'preprocess/SP_FT/s2wrapper.py', Path(__file__)]}}
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        feature_root = out / 'features' / dirname
        marker = feature_root / 'extraction.json'
        if marker.exists() and json.loads(marker.read_text())['fingerprint'] != fingerprint:
            if not args.force:
                raise ValueError(f'{feature}: incompatible extraction settings/source; use a new output-root or --force to quarantine old features')
            feature_root.rename(feature_root.with_name(dirname + '.stale-' + str(time.time_ns())))
        elif feature_root.exists() and not marker.exists() and any(feature_root.rglob('*.npy')):
            raise ValueError(f'{feature_root}: untracked features; choose a fresh output-root')
        _json(marker, {'fingerprint': fingerprint, 'config': config})
        anno = out / 'extraction_jobs' / fingerprint / 'How2Sign'
        anno.mkdir(parents=True, exist_ok=True)
        for source_split in splits:
            target_split = 'dev' if source_split == 'val' else source_split
            records = [s for s in source['samples'] if s['source_split'] == source_split]
            if args.limit:
                records = records[:args.limit]
            if not records:
                raise ValueError(f'No samples for {source_split}')
            pending = []
            for record in records:
                path = feature_root / target_split / (record['fileid'] + '.npy')
                if path.exists():
                    if not (args.resume or args.force):
                        raise FileExistsError(f'{path}: use --resume or --force')
                    try:
                        shape = check_array(np.load(path, allow_pickle=False), width, record['fileid'])
                    except (ValueError, OSError, EOFError):
                        path.rename(path.with_suffix('.corrupt-' + str(time.time_ns())))
                    else:
                        reports.append({'id': record['fileid'], 'feature': feature, 'shape': shape, 'status': 'reused'})
                        continue
                pending.append(record)
            if not pending:
                continue
            manifest = {'prefix': ''}
            manifest.update(enumerate(pending))
            from .npyio import write_object
            write_object(anno / f'{source_split}_info.npy', manifest)
            opts = SimpleNamespace(anno_root=str(anno), video_root=str(args.dataset_root or ''), save_dir=str(out/'features'), device=args.device,
                model_name=model, batch_size=args.batch_size, cache_dir=cache, nth_layer=-1, overlap_size=8, s2_mode='s2wrapping', scales=[1,2])
            target_dir = feature_root / target_split
            target_dir.mkdir(parents=True, exist_ok=True)
            original_resume = module.find_resume_batch
            module.read_video = checked_read
            module.find_resume_batch = checked_resume_factory(module, width)
            expected = {r['fileid'] for r in pending}
            completed = set()
            try:
                iterator, count = module.get_iterator(opts, source_split, save_mode=target_split)
                if count != len(pending):
                    raise ValueError('Extractor count differs from manifest count')
                for array, fileid, _ in iterator():
                    if fileid not in expected or fileid in completed:
                        raise ValueError(f'Unexpected or duplicate extracted ID: {fileid}')
                    completed.add(fileid)
                    array = np.asarray(array)
                    shape = check_array(array, width, fileid)
                    module.atomic_save_npy(str(target_dir / f'{fileid}.npy'), array)
                    reports.append({'id': fileid, 'feature': feature, 'shape': shape, 'status': 'extracted'})
                    _json(out/'extraction_report.json', {'samples': reports, 'fingerprint': fingerprint})
                if completed != expected:
                    raise ValueError(f'Extractor omitted IDs: {sorted(expected - completed)[:10]}')
            except (OSError, ValueError, RuntimeError) as exc:
                _json(out/'extraction_report.json', {'samples': reports, 'error': str(exc), 'feature': feature, 'split': source_split})
                raise RuntimeError(f'{feature}/{source_split}: {exc}. Models must already be cached; extraction is offline.') from exc
            finally:
                module.read_video = original_read
                module.find_resume_batch = original_resume
    _json(out/'extraction_report.json', {'samples': reports, 'plan': plan})
    return {'samples': reports, 'plan': plan}


def dependency_versions():
    import importlib.metadata
    result = {}
    for name in ('torch', 'transformers', 'numpy', 'av', 'Pillow'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'unavailable'
    return result


def assert_source_unchanged(samples):
    for sample in samples:
        fileid = sample['fileid']
        if not fileid or any(c in fileid for c in './\\:'):
            raise ValueError(f'Unsafe sample ID: {fileid!r}')
        path = Path(sample['folder'])
        if not path.is_file():
            raise ValueError(f'{fileid}: missing source video {path}')
        actual = {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        if sample.get('source_stat') != actual:
            raise ValueError(f'{fileid}: source video changed; prepare into a new output-root')


def checked_resume_factory(module, width):
    def checked_resume(checkpoint_dir, num_batches):
        import numpy as np
        for i in range(num_batches):
            path = Path(module.get_batch_checkpoint_path(checkpoint_dir, i))
            try:
                check_array(np.load(path, allow_pickle=False), width, str(path))
            except (OSError, EOFError, ValueError):
                for later in Path(checkpoint_dir).glob('batch_*.npy'):
                    try:
                        index = int(later.stem[6:])
                    except ValueError:
                        continue
                    if index >= i:
                        later.rename(later.with_suffix('.invalid-' + str(time.time_ns())))
                return i
        return num_batches
    return checked_resume


def extract(args):
    if args.dry_run:
        return _extract(args)
    out = Path(args.output_root).resolve()
    repo = Path(args.repo_root).resolve()
    if out == repo or repo in out.parents or out in repo.parents:
        raise ValueError('output-root must be separate from the repository')
    source_path = out/'source_manifest.json'
    if not source_path.is_file():
        raise ValueError(f'Missing {source_path}; run prepare first')
    source = json.loads(source_path.read_text(encoding='utf-8-sig'))
    if source.get('synthetic'):
        raise ValueError('Synthetic artifacts cannot be used for real extraction')
    raw = Path(source['dataset_root']).resolve()
    if out == raw or raw in out.parents or out in raw.parents:
        raise ValueError('output-root must be separate from raw data')
    with (out/'.extraction.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another extraction owns {out}') from exc
        try:
            return _extract(args)
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
