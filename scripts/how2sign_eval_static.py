#!/usr/bin/env python3
"""Standard-library acceptance checks. Does not download or import ML models."""
import argparse
import ast
import compileall
import json
import os
from pathlib import Path
import sys

REQUIRED = {'dataset_root','dict_path','feature_dir','sp_path','mo_path','phm_path','pose_path','pose_encoder_config_path','language'}
CHANGED = ['main.py','test.py','ablation_visual.py','dataset/loader_ft.py','models/edm_ddp.py']


def _is_portable(value, roots):
    """A path is portable if it is relative or resolves under a selected root.

    Generated configs legitimately contain absolute paths under the user-chosen
    output root (which may itself live under a home directory, e.g. C:/Users/...).
    Baked-in developer paths (an absolute path outside the selected roots) are
    rejected instead of substring-matching '/home/' or '/Users/'.
    """
    if not isinstance(value, str) or ('/' not in value and '\\' not in value):
        return True
    expanded = Path(os.path.expandvars(value)).expanduser()
    if not expanded.is_absolute():
        return True
    for root in roots:
        if root and (expanded == root or root in expanded.parents):
            return True
    return False


def evaluate(repo, output):
    package = Path(__file__).resolve().parents[1]
    checks = []
    def check(name, action):
        try:
            action()
        except (AssertionError, OSError, ValueError, KeyError, StopIteration, SyntaxError) as exc:
            checks.append((name, False, str(exc)))
        else:
            checks.append((name, True, ''))
    def compilation():
        assert compileall.compile_dir(str(package/'how2sign_pipeline'),quiet=1)
        assert compileall.compile_dir(str(package/'scripts'),quiet=1)
        assert compileall.compile_dir(str(package/'tests'),quiet=1)
        for filename in CHANGED:
            assert compileall.compile_file(str(repo/filename),quiet=1), filename
    check('compiled: external package and five integration modules', compilation)
    def configurations():
        source = (output/'configs/How2Sign.yaml').read_text(encoding='utf-8-sig')
        data = json.loads(source)  # Generated JSON is valid YAML 1.2; no YAML dependency needed.
        assert REQUIRED <= data.keys(), str(REQUIRED-data.keys())
        roots = [Path(output), Path(data['dataset_root']).expanduser() if isinstance(data.get('dataset_root'), str) else None]
        for key, value in data.items():
            assert _is_portable(value, roots), f'{key} is a baked-in absolute path: {value!r}'
        run = json.loads((output/'configs/how2sign_edm.yaml').read_text(encoding='utf-8-sig'))
        assert run['dataset'] == 'How2Sign' and run['include_ctc'] is False
        assert run['include_pose'] is False and data['language'].lower() in ('english','en')
        assert all(isinstance(run[k],int) and run[k]>0 for k in ['sp_hidden_size','mo_hidden_size'])
    check('static-checked: dataset and run config keys', configurations)
    def discovery():
        source = (repo/'dataset/loader_ft.py').read_text(encoding='utf-8-sig')
        tree=ast.parse(source)
        assert 'fname.split(".")[0]' not in source and "fname.split('.')[0]" not in source
        init=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        assignments=[n for n in ast.walk(init) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='files' for t in n.targets)]
        assert any(isinstance(n.value,ast.ListComp) and any(isinstance(x,ast.Name) and x.id=='annots' for x in ast.walk(n.value)) for n in assignments), 'self.files must derive from annots'
    check('static-checked: manifest discovery and period-safe IDs', discovery)
    def splits():
        tree=ast.parse((repo/'main.py').read_text(encoding='utf-8-sig'))
        setup=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='setup_data')
        branches=[n for n in ast.walk(setup) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=='how2sign' for x in ast.walk(n.test))]
        assert branches, 'How2Sign branch absent'
        assert any(isinstance(x,ast.List) and [v.value for v in x.elts if isinstance(v,ast.Constant)]==['train','dev','test'] for x in ast.walk(branches[0]))
    check('static-checked: How2Sign train/dev/test selection', splits)
    def prompt():
        tree=ast.parse((repo/'models/edm_ddp.py').read_text(encoding='utf-8-sig'))
        model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EDM')
        forward=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        prompts=[n for n in ast.walk(forward) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompts' for t in n.targets)]
        assert prompts, 'No prompts assignment found'
        for assignment in prompts:
            assert not any(isinstance(n,ast.Name) and n.id in {'text','texts','english_text','english_texts'} for n in ast.walk(assignment.value)), 'Target used in prompts'
            # Only loaded names matter; names introduced in a Store context (e.g. the
            # comprehension variable `_`) are bindings, not dependencies.
            loaded = {n.id for n in ast.walk(assignment.value) if isinstance(n,ast.Name) and isinstance(n.ctx, ast.Load)}
            assert loaded <= {'self','bs','range'}, f'Unexpected prompt dependency: {sorted(loaded - {"self","bs","range"})}'
    check('static-checked: instruction prompt excludes target variables', prompt)
    for name, passed, detail in checks:
        print(('PASS ' if passed else 'FAIL ') + name + (': '+detail if detail else ''))
    return 0 if all(passed for _,passed,_ in checks) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',default=os.environ.get('VIPOSLT_REPO'),required=not bool(os.environ.get('VIPOSLT_REPO')))
    parser.add_argument('--output-root',required=True)
    args=parser.parse_args()
    return evaluate(Path(args.repo_root).resolve(),Path(args.output_root).resolve())
if __name__=='__main__': sys.exit(main())
