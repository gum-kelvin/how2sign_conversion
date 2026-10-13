"""Offline contract tests. Loader runs its real code with explicitly mocked tensors."""
import ast
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(os.environ.get('VIPOSLT_REPO', Path(__file__).resolve().parents[2] / 'serverViPo-SLT'))

class Tensor:
    def __init__(self, values):
        self.values = values.values if isinstance(values, Tensor) else values
        def shape(v):
            return (len(v),) + shape(v[0]) if isinstance(v, (list, tuple)) and v else ()
        self.shape = shape(self.values)
    def __iter__(self):
        return iter(self.values)


def padded(values, batch_first=True):
    longest = max(v.shape[0] for v in values)
    return Tensor([v.values + [[0.0] * v.shape[1] for _ in range(longest-v.shape[0])] for v in values])


def dependency_mocks(registry):
    names = ('torch', 'torch.utils', 'torch.utils.data', 'torch.nn', 'torch.nn.utils', 'torch.nn.utils.rnn', 'torch.nn.functional', 'numpy')
    modules = {name: types.ModuleType(name) for name in names}
    modules['torch'].tensor = lambda values, dtype=None: Tensor(values)
    modules['torch'].float32 = 'float32'
    modules['torch'].long = 'long'
    modules['torch.utils.data'].Dataset = object
    modules['torch.nn.utils.rnn'].pad_sequence = padded
    def load(path, allow_pickle=False):
        obj = registry[str(Path(path).resolve())]
        return types.SimpleNamespace(item=lambda: obj) if allow_pickle else obj
    modules['numpy'].load = load
    return modules


class LoaderRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cwd = Path.cwd()
        os.chdir(self.root)
        self.registry = {}
        self.mocks = patch.dict('sys.modules', dependency_mocks(self.registry))
        self.mocks.start()
        spec = importlib.util.spec_from_file_location('actual_loader', REPO/'dataset/loader_ft.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.loader = mod.PrecomputedFeatureDataset
    def tearDown(self):
        self.mocks.stop()
        os.chdir(self.cwd)
        self.temp.cleanup()
    def fixture(self, dataset='How2Sign', ids=('h61','h62'), sp=True, mo=True):
        annots = {'prefix': ''}
        for i, fileid in enumerate(ids):
            text = 'Official English, punctuation | café ' + str(i)
            annots[i] = {'fileid':fileid, 'text':text, 'en_text':text, 'gloss':''}
            for modality, enabled, width in [('sp',sp,4),('mo',mo,3)]:
                if enabled:
                    path = self.root/modality/(fileid+'.npy')
                    path.parent.mkdir(exist_ok=True)
                    path.touch()
                    self.registry[str(path)] = [[0.25]*width for _ in range(i+2)]
        self.registry[str(self.root/'preprocess'/dataset/'train_info_ml.npy')] = annots
        return self.loader(str(self.root/'no-frame-sentinels'), str(self.root/'sp'), str(self.root/'no-pose'), str(self.root/'mo'), dataset=dataset, include_sp=sp, include_mo=mo, include_pose=False)
    def test_how2sign_actual_loader_batch_mocked_tensor_backend(self):
        ds = self.fixture()
        self.assertEqual(ds.files, ['h61','h62'])
        batch = ds.collate_fn([ds[0],ds[1]])
        self.assertEqual(batch[0].shape, (2,3,4))
        self.assertEqual(batch[2].shape, (2,3,3))
        self.assertEqual(batch[6].values,[2,3])
        self.assertEqual(batch[8].values,[2,3])
        self.assertEqual(batch[4],batch[5])
        self.assertEqual(batch[3], ('',''))
        self.assertEqual(batch[0].values[0][-1],[0.0]*4)
        self.assertTrue(all(math.isfinite(x) for row in batch[0].values for frame in row for x in frame))
    def test_phoenix_period_ids_and_manifest_prefix(self):
        ds = self.fixture('phoenix2014-T', ('weather.001','weather.002'))
        self.assertEqual(len(ds),2)
        self.assertEqual(ds[0][4], ds[0][5])
        self.assertEqual(ds.files[0], 'weather.001')
    def test_disabling_spatial_does_not_require_spatial(self):
        ds = self.fixture(sp=False)
        self.assertIsNone(ds[0][0])
        self.assertEqual(ds.collate_fn([ds[0],ds[1]])[2].shape,(2,3,3))
    def test_disabling_motion_does_not_require_motion(self):
        ds = self.fixture(mo=False)
        self.assertIsNone(ds[0][2])
    def test_strict_missing_features(self):
        self.fixture()
        (self.root/'sp/h61.npy').unlink()
        with self.assertRaisesRegex(ValueError,'h61'):
            self.loader('unused',str(self.root/'sp'),'unused',str(self.root/'mo'),dataset='How2Sign',include_pose=False)
    def test_duplicate_manifest_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.fixture(ids=('h61','h61'))
    def test_how2sign_period_id_rejected(self):
        with self.assertRaisesRegex(ValueError,'period'):
            self.fixture(ids=('unsafe.id',))


class PromptRegression(unittest.TestCase):
    def test_actual_forward_prompt_expression_excludes_targets(self):
        tree = ast.parse((REPO/'models/edm_ddp.py').read_text(encoding='utf-8'))
        model = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EDM')
        forward = next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        assignment = next(n for n in forward.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompts' for t in n.targets))
        expression = ast.Expression(assignment.value)
        target = 'UNIQUE target sentence must never leak into instruction'
        prompts = eval(compile(expression,'actual-model-prompt','eval'), {'self':types.SimpleNamespace(prompt='Translate signed video into English'), 'bs':2,'texts':[target,target],'english_texts':[target,target]})
        self.assertEqual(len(prompts),2)
        self.assertTrue(all(target not in prompt for prompt in prompts))
        self.assertFalse(any(isinstance(n,ast.Name) and n.id in {'texts','english_texts','text'} for n in ast.walk(assignment.value)))


class MetadataTests(unittest.TestCase):
    def test_id_reversibility_unicode_and_periods(self):
        from how2sign_pipeline.metadata import safe_id
        ids = ['a.b','a/b','a_b','café','a b']
        encoded = [safe_id(x) for x in ids]
        self.assertEqual(len(set(encoded)),len(ids))
        for original, encoded_id in zip(ids,encoded):
            self.assertNotIn('.',encoded_id)
            self.assertEqual(bytes.fromhex(encoded_id[1:]).decode('utf8'),original)
    def test_invalid_timestamps(self):
        from how2sign_pipeline.metadata import parse_timestamp
        for value in ['nan','inf','-1','oops']:
            with self.subTest(value=value),self.assertRaises(ValueError): parse_timestamp(value)
        self.assertEqual(parse_timestamp(' 1.25 '),1.25)
    def test_official_headers_csv_tsv_preserve_sentence(self):
        from how2sign_pipeline.metadata import read_annotations
        for delimiter in [',','\t']:
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/('train.csv' if delimiter==',' else 'train.tsv')
                sentence='  Café, "hello" | welcome! 雨  '
                with path.open('w',encoding='utf8',newline='') as handle:
                    writer=csv.writer(handle,delimiter=delimiter)
                    writer.writerow(['SENTENCE_NAME','VIDEO_NAME','START_REALIGNED','END_REALIGNED','SENTENCE'])
                    writer.writerow(['sentence.1','video',1,2,sentence])
                rows=read_annotations(path,'val')
                self.assertEqual(rows[0]['text'],sentence)
                self.assertEqual(rows[0]['en_text'],sentence)
                self.assertEqual(rows[0]['gloss'],'')
                self.assertEqual(rows[0]['split'],'dev')
    def test_missing_fields_fail(self):
        from how2sign_pipeline.metadata import read_annotations
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'train.csv'; path.write_text('SENTENCE_ID,SENTENCE\nx,hello\n')
            with self.assertRaises(ValueError): read_annotations(path,'train')
    def test_npy_float_wire_format_and_corruption(self):
        from how2sign_pipeline.npyio import write_float,read_float
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'sample.npy'
            write_float(path,2,4)
            self.assertEqual(path.read_bytes()[:6],b'\x93NUMPY')
            result=read_float(path)
            self.assertEqual(tuple(result['shape']),(2,4))
            self.assertTrue(result['finite'])
            path.write_bytes(b'broken')
            with self.assertRaises((ValueError,EOFError)): read_float(path)

if __name__ == '__main__': unittest.main()
