"""Exercise the repository's real dataset/collator with explicit stdlib mocks.

The mocked backend loads generated numeric NPY bytes, and JSON annotation sidecars.
No invented feature is represented as a real extraction. Use real_dependencies=True
only when numpy and torch are installed to test the same batch without mocks.
"""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import types
from unittest.mock import patch


class Tensor:
    def __init__(self, values):
        self.values = values.values if isinstance(values, Tensor) else values
        def shape(value):
            return (len(value),) + shape(value[0]) if isinstance(value, (list, tuple)) and value else ()
        self.shape = shape(self.values)
    def __iter__(self):
        return iter(self.values)
    def tolist(self):
        return self.values


def padded(values, batch_first=True):
    longest = max(v.shape[0] for v in values)
    return Tensor([v.values + [[0.0] * v.shape[1] for _ in range(longest-v.shape[0])] for v in values])


def read_numeric(path):
    from .npyio import read_float
    info = read_float(path)
    with Path(path).open('rb') as stream:
        stream.read(6)
        version = stream.read(2)
        size = int.from_bytes(stream.read(2 if version[0] == 1 else 4), 'little')
        header = ast.literal_eval(stream.read(size).decode('latin1'))
        dtype = header['descr']
        flat = [v[0] for v in struct.iter_unpack(dtype[0] + ('f' if dtype[-1] == '4' else 'd'), stream.read())]
    rows, width = info['shape']
    return [flat[i*width:(i+1)*width] for i in range(rows)]


def dependency_mocks():
    names = ('torch', 'torch.utils', 'torch.utils.data', 'torch.nn', 'torch.nn.utils', 'torch.nn.utils.rnn', 'torch.nn.functional', 'numpy')
    modules = {name: types.ModuleType(name) for name in names}
    modules['torch'].tensor = lambda values, dtype=None: Tensor(values)
    modules['torch'].float32 = 'float32'
    modules['torch'].long = 'long'
    modules['torch.utils.data'].Dataset = object
    modules['torch.nn.utils.rnn'].pad_sequence = padded
    def load(path, allow_pickle=False):
        path = Path(path)
        if allow_pickle:
            sidecar = path.with_suffix('.json')
            value = json.loads(sidecar.read_text(encoding='utf8'))
            value = {int(key) if key.isdigit() else key: record for key, record in value.items()}
            return types.SimpleNamespace(item=lambda: value)
        return read_numeric(path)
    modules['numpy'].load = load
    return modules


def run(repo_root, output_root, real_dependencies=False):
    repo_root, output_root = Path(repo_root).resolve(), Path(output_root).resolve()
    info = json.loads((output_root/'configs/How2Sign.yaml').read_text(encoding='utf8'))
    config = json.loads((output_root/'configs/how2sign_edm.yaml').read_text(encoding='utf8'))
    backend = 'real numpy/torch' if real_dependencies else 'explicit stdlib numpy/torch mocks'
    reports = []
    def check():
        spec = importlib.util.spec_from_file_location('_how2sign_actual_loader', repo_root/'dataset/loader_ft.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for mode in ('train', 'dev', 'test'):
            annot_path = output_root/'preprocess/How2Sign'/f'{mode}_info_ml.json'
            if not annot_path.exists():
                continue
            annotations = json.loads(annot_path.read_text(encoding='utf8'))
            records = [record for key, record in annotations.items() if key.isdigit()]
            if not records:
                continue
            dataset = module.PrecomputedFeatureDataset(
                str(Path(info['feature_dir'])/mode), str(Path(info['sp_path'])/mode),
                info['pose_path'], str(Path(info['mo_path'])/mode), dataset='How2Sign', mode=mode,
                include_sp=config['include_sp'], include_pose=False, include_mo=config['include_mo'], strict=True)
            expected_ids = [record['fileid'] for record in records]
            assert dataset.files == expected_ids, (dataset.files, expected_ids)
            count = min(2, len(dataset))
            samples = [dataset[index] for index in range(count)]
            assert all(len(sample) == 6 for sample in samples)
            batch = dataset.collate_fn(samples)
            expected_text = tuple(record['text'] for record in records[:count])
            assert batch[4] == expected_text and batch[5] == expected_text
            assert all(value == '' for value in batch[3])
            assert batch[1] is None and batch[7] is None
            shapes = {}
            for modality, index, length_index, width_key in [('clip',0,6,'sp_hidden_size'),('videomae',2,8,'mo_hidden_size')]:
                if batch[index] is None:
                    continue
                tensor = batch[index]
                values = tensor.tolist()
                lengths = batch[length_index].tolist()
                expected_lengths = [sample[index].shape[0] for sample in samples]
                assert lengths == expected_lengths
                assert tuple(tensor.shape) == (count, max(lengths), config[width_key])
                assert all(math.isfinite(value) for sample in values for frame in sample for value in frame)
                for sample, length in zip(values, lengths):
                    assert all(value == 0.0 for frame in sample[length:] for value in frame)
                shapes[modality] = list(tensor.shape)
            reports.append({'split':mode, 'ids':dataset.files[:count], 'batch_shapes':shapes, 'targets_match':True, 'padding_verified':True})
    previous = Path.cwd()
    try:
        os.chdir(output_root)
        if real_dependencies:
            check()
        else:
            with patch.dict('sys.modules', dependency_mocks()):
                check()
    finally:
        os.chdir(previous)
    if not reports:
        raise ValueError('No generated split manifests available for loader smoke test')
    return {'backend':backend, 'actual_repository_loader':str(repo_root/'dataset/loader_ft.py'), 'batches':reports,
            'verification':'real-dependency fixture' if real_dependencies else 'stdlib-fixture-tested'}
