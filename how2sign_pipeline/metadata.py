"""Read-only, indexed How2Sign annotation and media inventory."""
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

SPLITS = {'train': 'train', 'val': 'dev', 'test': 'test'}
REQUIRED = ('VIDEO_NAME', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE')
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def safe_id(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Empty or non-string sample ID')
    encoded = 'h' + value.encode('utf-8').hex()
    if len(encoded) > 240:
        raise ValueError(f'Sample ID exceeds portable filename length: {value!r}')
    return encoded


def parse_timestamp(value, name='timestamp'):
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name}: expected numeric seconds, got {value!r}') from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'{name}: expected finite nonnegative seconds, got {value!r}')
    return number


def read_annotations(path, split):
    if split not in SPLITS:
        raise ValueError(f'Unsupported source split {split}; use train,val,test')
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters='\t,;|')
        except csv.Error:
            dialect = csv.excel_tab if '\t' in sample.partition('\n')[0] else csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        normalized = [str(x).strip().upper() for x in (reader.fieldnames or [])]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f'{path}: duplicate normalized column names')
        missing = set(REQUIRED) - set(normalized)
        if missing:
            raise ValueError(f'{path}: missing required columns: {sorted(missing)}; found {normalized}')
        records, seen = [], set()
        for line, raw in enumerate(reader, 2):
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError(f'{path}:{line}: malformed delimited row')
            row = {str(key).strip().upper(): val for key, val in raw.items()}
            sid = row['SENTENCE_NAME']
            if sid in seen:
                raise ValueError(f'{path}:{line}: duplicate sample ID {sid!r}')
            seen.add(sid)
            start = parse_timestamp(row['START_REALIGNED'], f'{sid} START_REALIGNED')
            end = parse_timestamp(row['END_REALIGNED'], f'{sid} END_REALIGNED')
            if start >= end:
                raise ValueError(f'{sid}: require START_REALIGNED < END_REALIGNED')
            if not row['SENTENCE'].strip() or not row['VIDEO_NAME'].strip():
                raise ValueError(f'{path}:{line}: empty SENTENCE or VIDEO_NAME')
            info = dict(row)
            info.update(START_REALIGNED=start, END_REALIGNED=end)
            records.append(dict(fileid=safe_id(sid), source_id=sid, source_video=row['VIDEO_NAME'],
                                source_split=split, split=SPLITS[split], start=start, end=end,
                                text=row['SENTENCE'], en_text=row['SENTENCE'], gloss='',
                                raw_metadata=dict(raw), original_info=info,
                                signer=row.get('SIGNER_ID') or row.get('SIGNER') or None,
                                annotation_path=str(path.resolve())))
    if not records:
        raise ValueError(f'{path}: annotation table contains no samples')
    return records


def _stem(name):
    path = Path(name)
    return path.stem if path.suffix.lower() in VIDEO_SUFFIXES else path.name


def _camera(path):
    value = str(path).lower().replace('\\', '/')
    if 'rgb_side' in value or '/side/' in value:
        return 'side'
    if 'rgb_front' in value or '/front/' in value:
        return 'front'
    return 'unknown'


def _matches(index, name, split, clip=False):
    stem = _stem(name)
    candidates = set()
    for key in (stem, stem + '-rgb_front', stem + '_rgb_front'):
        candidates.update(index.get(key, ()))
    candidates = [p for p in candidates if _camera(p) != 'side']
    candidates = [p for p in candidates if not (set(p.parts) & set(SPLITS) - {split})]
    if not clip:
        candidates = [p for p in candidates if 'sentence_level' not in p.parts]
    front = [p for p in candidates if _camera(p) == 'front']
    return sorted(front or candidates)


def probe_video(path, executable='ffprobe'):
    command = [executable, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=width,height,duration,avg_frame_rate:format=duration', '-of', 'json', str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError(f'{path}: ffprobe failed: {result.stderr.strip()}')
    data = json.loads(result.stdout)
    streams = data.get('streams') or []
    if not streams:
        raise ValueError(f'{path}: no video stream')
    stream = streams[0]
    duration = stream.get('duration') or data.get('format', {}).get('duration')
    measured = parse_timestamp(duration, f'{path} duration') if duration not in (None, 'N/A') else None
    return {'duration': measured, 'width': stream.get('width'), 'height': stream.get('height'),
            'avg_frame_rate': stream.get('avg_frame_rate')}


def inspect_dataset(dataset_root, splits=('train', 'val', 'test'), allow_sentence_clips=False, probe=False):
    root = Path(os.path.expandvars(str(dataset_root))).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset root does not exist: {root}')
    if probe and not shutil.which('ffprobe'):
        raise ValueError('--probe requires ffprobe on PATH; omit it for metadata-only inspection')
    files = sorted(p for p in root.rglob('*') if p.is_file())
    videos = [p for p in files if p.suffix.lower() in VIDEO_SUFFIXES]
    index, resource_index = {}, {}
    for path in files:
        resource_index.setdefault(path.stem, []).append(str(path.resolve()))
    for video in videos:
        index.setdefault(video.stem, []).append(video)
    samples, errors, metadata, seen, probes = [], [], [], set(), {}
    columns = {}
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f'Unknown source split {split}; use train,val,test')
        matches = [p for p in files if p.suffix.lower() in {'.csv', '.tsv'} and
                   (p.stem.lower() in {split, f'{split}_info', f'how2sign_realigned_{split}', f'how2sign_{split}'}
                    or f'_{split}_' in p.stem.lower())]
        realigned = [p for p in matches if 'realigned' in p.stem.lower()]
        matches = realigned or matches
        if len(matches) != 1:
            errors.append(f'{split}: expected one annotation table, found {len(matches)}: {[str(p) for p in matches]}')
            continue
        metadata.append(str(matches[0]))
        try:
            records = read_annotations(matches[0], split)
        except (ValueError, csv.Error, UnicodeError) as exc:
            errors.append(str(exc))
            continue
        columns[split] = list(records[0]['raw_metadata'])
        for record in records:
            sid = record['source_id']
            if sid in seen:
                errors.append(f'Duplicate sample ID across splits: {sid}')
                continue
            seen.add(sid)
            candidates = _matches(index, record['source_video'], split)
            fallback = False
            if not candidates and allow_sentence_clips:
                candidates = _matches(index, sid, split, clip=True)
                fallback = bool(candidates)
            if len(candidates) != 1:
                errors.append(f'{sid}: {"missing" if not candidates else "ambiguous"} frontal source video: {[str(p) for p in candidates]}')
                continue
            info = record['original_info']
            try:
                if fallback:
                    if info.get('START') in (None, '') or info.get('END') in (None, ''):
                        raise ValueError('sentence clip fallback requires original START and END to derive relative realigned crop')
                    origin, stop = parse_timestamp(info['START']), parse_timestamp(info['END'])
                    start, end = record['start'] - origin, record['end'] - origin
                    if start < 0 or start >= end or end > stop - origin:
                        raise ValueError('realigned crop lies outside sentence clip; use original full video')
                    info['SOURCE_START_REALIGNED'], info['SOURCE_END_REALIGNED'] = record['start'], record['end']
                    info['START_REALIGNED'], info['END_REALIGNED'] = start, end
                path = candidates[0]
                if probe and path not in probes:
                    probes[path] = probe_video(path)
                measurement = probes.get(path)
                if measurement and measurement['duration'] is not None and info['END_REALIGNED'] > measurement['duration']:
                    raise ValueError(f"END_REALIGNED {info['END_REALIGNED']} exceeds video duration {measurement['duration']}; no automatic crop correction")
            except (ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                errors.append(f'{sid}: {exc}')
                continue
            record['folder'] = str(path.resolve())
            record['timing_reference'] = 'sentence_clip_relative' if fallback else 'source_video'
            record['video_probe'] = measurement
            resources = set()
            for name in (_stem(sid), _stem(record['source_video'])):
                for key in (name, name + '-rgb_front', name + '-rgb_side', name + '_rgb_front', name + '_rgb_side'):
                    resources.update(resource_index.get(key, ()))
            record['available_media'] = sorted(resources)
            record['source_stat'] = {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            samples.append(record)
    return {'dataset_root': str(root), 'samples': samples, 'errors': errors, 'metadata': metadata, 'columns': columns,
            'inventory': [str(p.relative_to(root)) for p in files],
            'split_counts': {split: sum(s['source_split'] == split for s in samples) for split in splits},
            'video_count': len(videos), 'probed_videos': len(probes),
            'duration_validation': 'Measured with ffprobe; strict bounds, zero tolerance' if probe else 'Not measured; use inspect --probe for duration/resolution checks',
            'active_modalities': ['rgb_front: CLIP spatial', 'rgb_front: VideoMAE motion'],
            'preserved_modalities': 'All source files indexed; side/pose/depth are not consumed',
            'estimated_feature_bytes_per_frame': 4 * (2048 + 1024)}
