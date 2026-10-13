"""Command line entry point for the external How2Sign pipeline.

Always resolves the repository explicitly (``--repo-root`` or ``VIPOSLT_REPO``)
and never assumes the current working directory is the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from . import artifacts, smoke
from .fixtures import build_synthetic_output
from .metadata import inspect_dataset

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def _repo(value):
    repo = Path(os.path.expandvars(str(value))).expanduser().resolve()
    if not (repo / 'main.py').is_file() or not (repo / 'dataset' / 'loader_ft.py').is_file():
        raise SystemExit(f'--repo-root does not look like the ViPo-SLT checkout: {repo}')
    return repo


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return str(path)


def cmd_inspect(args):
    report = inspect_dataset(args.dataset_root, splits=args.splits.split(','),
                             allow_sentence_clips=args.allow_sentence_clips, probe=args.probe)
    summary = {'dataset_root': report['dataset_root'], 'samples': len(report['samples']),
               'errors': report['errors'], 'splits': report['split_counts'],
               'columns': report['columns'], 'active_modalities': report['active_modalities'],
               'preserved_modalities': report['preserved_modalities']}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        print('Wrote', _write_report(args.report, report))
    if report['errors']:
        print(f'inspection reported {len(report["errors"])} error(s)', file=sys.stderr)
        return EXIT_FAIL if args.strict else EXIT_OK
    return EXIT_OK


def cmd_prepare(args):
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding='utf-8-sig'))
    else:
        report = inspect_dataset(args.dataset_root, splits=args.splits.split(','),
                                 allow_sentence_clips=args.allow_sentence_clips, probe=args.probe)
    summary = artifacts.prepare(report, args.output_root, repo_root=args.repo_root,
                                force=args.force, pilot=args.pilot)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return EXIT_OK


def _extraction_namespace(args, dataset_root):
    return SimpleNamespace(repo_root=args.repo_root, output_root=args.output_root,
                           dataset_root=dataset_root, features=args.features, splits=args.splits,
                           batch_size=args.batch_size, limit=args.limit, device=args.device,
                           dry_run=args.dry_run, force=args.force, resume=args.resume,
                           cache_dir=args.cache_dir)


def cmd_extract(args):
    from .extraction import extract
    source = Path(args.output_root) / 'source_manifest.json'
    dataset_root = args.dataset_root
    if dataset_root is None and source.is_file():
        dataset_root = json.loads(source.read_text(encoding='utf-8-sig')).get('dataset_root')
    result = extract(_extraction_namespace(args, dataset_root))
    print(json.dumps({'plan': result.get('plan'), 'samples': len(result.get('samples', []))},
                     ensure_ascii=False, indent=2, default=str))
    return EXIT_OK


def cmd_validate(args):
    report = artifacts.validate(args.output_root, repo_root=args.repo_root, strict=args.strict,
                                require_features=not args.no_features)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return EXIT_OK if report['ok'] else EXIT_FAIL


def cmd_all(args):
    report = inspect_dataset(args.dataset_root, splits=args.splits.split(','),
                             allow_sentence_clips=args.allow_sentence_clips, probe=args.probe)
    artifacts.prepare(report, args.output_root, repo_root=args.repo_root, force=args.force,
                      pilot=args.pilot)
    from .extraction import extract
    extract(_extraction_namespace(args, report['dataset_root']))
    result = artifacts.validate(args.output_root, repo_root=args.repo_root, strict=True,
                                require_features=not args.no_features)
    print(json.dumps({'prepared': True, 'validated': result['ok'], 'splits': result['splits']},
                     ensure_ascii=False, indent=2, default=str))
    return EXIT_OK if result['ok'] else EXIT_FAIL


def cmd_smoke(args):
    repo = args.repo_root
    output = Path(args.output_root).expanduser().resolve() if args.output_root else \
        Path(tempfile.mkdtemp(prefix='how2sign-smoke-'))
    try:
        build_synthetic_output(output, repo_root=repo, force=True)
        loader = smoke.run(repo, output)
        validation = artifacts.validate(output, repo_root=repo, strict=True)
        result = {'output_root': str(output), 'loader': loader, 'validation': validation,
                  'verification': 'stdlib-fixture-tested'}
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return EXIT_OK
    finally:
        if not args.output_root and not args.keep:
            shutil.rmtree(output, ignore_errors=True)


def build_parser():
    parser = argparse.ArgumentParser(prog='how2sign_pipeline', description=__doc__)
    parser.add_argument('--version', action='version', version='how2sign_pipeline 0.2.0')
    sub = parser.add_subparsers(dest='command', required=True)

    def common(p, repo=False, output=True):
        if repo:
            p.add_argument('--repo-root', default=os.environ.get('VIPOSLT_REPO'), required=not os.environ.get('VIPOSLT_REPO'))
        if output:
            p.add_argument('--output-root', default=os.environ.get('VIPOSLT_FEATURE_ROOT'))

    def inspect_opts(p):
        p.add_argument('--dataset-root', default=os.environ.get('HOW2SIGN_ROOT'))
        p.add_argument('--splits', default='train,val,test')
        p.add_argument('--probe', action='store_true')
        p.add_argument('--allow-sentence-clips', action='store_true', dest='allow_sentence_clips')
        p.add_argument('--report')
        p.add_argument('--strict', action='store_true')

    p = sub.add_parser('inspect', help='read-only metadata/media inventory')
    inspect_opts(p); p.set_defaults(func=cmd_inspect)

    p = sub.add_parser('prepare', help='build configs and manifests (no extraction)')
    p.add_argument('--dataset-root', default=os.environ.get('HOW2SIGN_ROOT'))
    p.add_argument('--splits', default='train,val,test')
    p.add_argument('--probe', action='store_true')
    p.add_argument('--allow-sentence-clips', action='store_true', dest='allow_sentence_clips')
    p.add_argument('--report'); p.add_argument('--force', action='store_true')
    p.add_argument('--pilot', action='store_true')
    common(p, repo=True); p.set_defaults(func=cmd_prepare)

    p = sub.add_parser('extract', help='run the existing CLIP/VideoMAE extractors')
    p.add_argument('--dataset-root', default=os.environ.get('HOW2SIGN_ROOT'))
    p.add_argument('--features', default='clip,videomae')
    p.add_argument('--splits', default='train,val,test')
    p.add_argument('--device', default='cuda:0'); p.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    p.add_argument('--limit', type=int, default=None); p.add_argument('--cache-dir', default=None, dest='cache_dir')
    for flag in ('resume', 'force', 'dry-run'):
        p.add_argument('--' + flag, action='store_true')
    common(p, repo=True); p.set_defaults(func=cmd_extract)

    p = sub.add_parser('validate', help='verify generated artifacts')
    p.add_argument('--no-features', action='store_true', dest='no_features',
                   help='skip feature coverage (manifests/configs only)')
    p.add_argument('--no-strict', action='store_false', dest='strict')
    p.set_defaults(strict=True)
    common(p, repo=True); p.set_defaults(func=cmd_validate)

    p = sub.add_parser('all', help='prepare -> extract -> strict validate')
    p.add_argument('--dataset-root', default=os.environ.get('HOW2SIGN_ROOT'))
    p.add_argument('--splits', default='train,val,test')
    p.add_argument('--probe', action='store_true')
    p.add_argument('--allow-sentence-clips', action='store_true', dest='allow_sentence_clips')
    p.add_argument('--features', default='clip,videomae')
    p.add_argument('--device', default='cuda:0'); p.add_argument('--batch-size', type=int, default=8, dest='batch_size')
    p.add_argument('--limit', type=int, default=None); p.add_argument('--cache-dir', default=None, dest='cache_dir')
    p.add_argument('--no-features', action='store_true', dest='no_features')
    for flag in ('resume', 'force', 'dry-run', 'pilot'):
        p.add_argument('--' + flag, action='store_true')
    common(p, repo=True); p.set_defaults(func=cmd_all)

    p = sub.add_parser('smoke-test', help='synthetic end-to-end loader smoke (stdlib only)')
    p.add_argument('--keep', action='store_true')
    common(p, repo=True); p.set_defaults(func=cmd_smoke)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    needs_repo = args.command in ('prepare', 'extract', 'validate', 'all', 'smoke-test')
    needs_output = args.command in ('prepare', 'extract', 'validate', 'all')
    if needs_repo:
        args.repo_root = _repo(args.repo_root)
    if needs_output and not args.output_root:
        parser.error(f'{args.command} requires --output-root or VIPOSLT_FEATURE_ROOT')
    if args.command in ('prepare', 'extract', 'all') and getattr(args, 'dataset_root', None) is None \
            and getattr(args, 'report', None) is None:
        parser.error(f'{args.command} requires --dataset-root, HOW2SIGN_ROOT or --report')
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        print(f'{args.command} failed: {exc}', file=sys.stderr)
        return EXIT_FAIL

