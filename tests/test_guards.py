"""Execute actual repository setup functions with bounded dependency stubs."""
import ast
import builtins
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, mock_open, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(os.environ.get('VIPOSLT_REPO', Path(__file__).resolve().parents[2]/'serverViPo-SLT'))


def extracted(filename, name, namespace):
    tree = ast.parse((REPO/filename).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REPO/filename), 'exec'), namespace)
    return namespace[name]


class RuntimeGateTests(unittest.TestCase):
    def call_setup(self, filename, ctc=False, pose=False, dataset='How2Sign'):
        arg = types.SimpleNamespace(optimizer_args={}, num_epoch=1, llm_args={}, llm='bigscience/mt0-large',
            dataset_info={'language':'English','dict_path':'missing-gloss.npy','pose_encoder_config_path':'missing-pose.yaml'},
            include_sp=True, include_pose=pose, include_mo=True, sp_hidden_size=2048, pose_hidden_size=512,
            mo_hidden_size=1024, contrastive=True, include_ctc=ctc, dataset=dataset,
            temporal_args={'num_heads':4,'num_layers':4}, feeder_ft='actual.loader')
        load = Mock(return_value=types.SimpleNamespace(item=lambda: {}))
        opener = mock_open(read_data='model: {}')
        namespace = {'os':os, 'np':types.SimpleNamespace(load=load), 'Logger':lambda _: lambda *a: None,
            'yaml':types.SimpleNamespace(safe_load=lambda _: {'model':{'RecognitionNetwork':{'verified':True}}}),
            'import_class':lambda _:object, 'shutil':types.SimpleNamespace(copy2=lambda *a:None),
            'inspect':types.SimpleNamespace(getfile=lambda _:'loader.py'), 'open':opener}
        name = 'build_llm_args' if filename == 'ablation_visual.py' else 'setup_work_dir'
        function = extracted(filename, name, namespace)
        if filename == 'main.py':
            function(arg, False, 'unused')
        elif filename == 'test.py':
            function(arg, 'unused')
        else:
            function(arg)
        return arg, load, opener

    def test_ctc_and_pose_disabled_do_not_load_assets(self):
        for filename in ('main.py','test.py','ablation_visual.py'):
            with self.subTest(filename=filename):
                arg, load, opener = self.call_setup(filename)
                self.assertEqual(arg.llm_args['gloss_dict'], {})
                self.assertEqual(arg.llm_args['pose_encoder_cfg'], {})
                load.assert_not_called(); opener.assert_not_called()

    def test_empty_gloss_ctc_rejected_in_all_entrypoints(self):
        for filename in ('main.py','test.py','ablation_visual.py'):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError,'genuine|verified gloss'):
                self.call_setup(filename, ctc=True)

    def test_how2sign_pose_rejected_in_all_entrypoints(self):
        for filename in ('main.py','test.py','ablation_visual.py'):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError,'pose'):
                self.call_setup(filename, pose=True)

    def test_phoenix_pose_configuration_still_loads(self):
        for filename in ('main.py','test.py','ablation_visual.py'):
            with self.subTest(filename=filename):
                arg, load, opener = self.call_setup(filename, pose=True, dataset='phoenix2014-T')
                opener.assert_called_once_with('missing-pose.yaml','r')
                self.assertEqual(arg.llm_args['pose_encoder_cfg'], {'verified':True})

    def test_actual_split_selection_preserves_existing_datasets(self):
        class Dataset:
            def __init__(self,*a,**kw): self.mode=kw['mode']; self.collate_fn=lambda x:x
        namespace={'os':os, 'import_class':lambda _:Dataset, 'DistributedSampler':lambda _:None,
            'torch':types.SimpleNamespace(utils=types.SimpleNamespace(data=types.SimpleNamespace(DataLoader=lambda ds,**kw:[ds.mode]))),
            'init_fn':lambda _:None}
        function=extracted('main.py','setup_data',namespace)
        for dataset,modes in [('CSL',['train','dev']),('CSL-Daily',['train','dev','test']),('phoenix2014-T',['train','dev','test']),('How2Sign',['train','dev','test'])]:
            with self.subTest(dataset=dataset):
                arg=types.SimpleNamespace(dataset=dataset, dataset_info={k:'unused' for k in ['feature_dir','sp_path','pose_path','mo_path']},
                    feeder_ft='actual.loader', include_sp=True,include_pose=False,include_mo=True,batch_size=2,test_batch_size=2,num_epoch=1)
                loaders,steps,samplers=function(arg,1,0,False)
                self.assertEqual(list(loaders),modes)

if __name__ == '__main__': unittest.main()
