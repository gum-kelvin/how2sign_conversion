"""Small NPY writer/inspector; never unpickle untrusted input.

Object writing uses the standard pickle numpy.array constructor, evaluated only by
NumPy when the trusted generated manifest is loaded. Offline readers use JSON.
"""
import ast
import math
import pickle
import struct
from pathlib import Path


def _header(descr, shape):
    raw = repr({'descr': descr, 'fortran_order': False, 'shape': shape}).encode('latin1')
    raw += b' ' * ((64 - (10 + len(raw) + 1) % 64) % 64) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(raw)) + raw


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def object_bytes(data):
    """Deterministic bytes for a rank-0 object NPY (trusted generated manifest)."""
    payload = b'\x80\x03cnumpy\narray\n' + pickle.dumps(data, protocol=3)[2:-1] + b'\x85R.'
    return _header('|O', ()) + payload


def write_object(path, data):
    atomic_bytes(path, object_bytes(data))


def float_bytes(rows, width, value=0.25):
    if rows <= 0 or width <= 0:
        raise ValueError('Synthetic feature dimensions must be positive')
    return _header('<f4', (rows, width)) + struct.pack('<f', value) * rows * width


def write_float(path, rows, width, value=0.25):
    atomic_bytes(path, float_bytes(rows, width, value))


def read_float(path):
    with Path(path).open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError(f'{path}: not a NPY file')
        version = stream.read(2)
        if version not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
            raise ValueError(f'{path}: unsupported NPY version')
        length_size = 2 if version[0] == 1 else 4
        length = int.from_bytes(stream.read(length_size), 'little')
        if length > 100000:
            raise ValueError(f'{path}: excessive NPY header')
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        dtype, shape = header['descr'], header['shape']
        if dtype not in ('<f4', '<f8', '>f4', '>f8') or len(shape) != 2 or any(x <= 0 for x in shape):
            raise ValueError(f'{path}: require nonempty rank-2 float array, got {dtype} {shape}')
        width = int(dtype[-1])
        count = math.prod(shape)
        remaining = count * width
        finite = True
        while remaining:
            chunk = stream.read(min(65536, remaining))
            if not chunk or len(chunk) % width:
                raise ValueError(f'{path}: truncated feature data')
            finite = finite and all(math.isfinite(v[0]) for v in struct.iter_unpack(dtype[0] + ('f' if width == 4 else 'd'), chunk))
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError(f'{path}: trailing feature bytes')
        if not finite:
            raise ValueError(f'{path}: nonfinite features')
    return {'shape': shape, 'dtype': dtype, 'finite': finite, 'count': count}
