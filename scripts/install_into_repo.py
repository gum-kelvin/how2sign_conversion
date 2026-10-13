"""Install a reversible bridge without copying pipeline sources into the repository."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import uuid

VERSION = 2
ARTIFACTS = ('configs/How2Sign.yaml', 'configs/how2sign_edm.yaml', 'preprocess/How2Sign')
PATCH_FILES = {'main.py', 'test.py', 'ablation_visual.py', 'dataset/loader_ft.py',
               'models/edm_ddp.py'}
MARKER = '.viposlt_integration.json'


def exists(path):
    return os.path.lexists(path)


def redirected(path):
    if not exists(path):
        return False
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def safe_path(path, leaf=False):
    """Reject redirected ancestors, including Windows junctions/reparse points."""
    path = Path(os.path.abspath(path))
    for item in reversed((path, *path.parents) if leaf else tuple(path.parents)):
        if redirected(item):
            raise RuntimeError(f'Refusing a symlink or reparse path: {item}')
    return path


def digest(path):
    """Hash names, types, empty directories, bytes, and symlink text without following links."""
    h = hashlib.sha256()
    def visit(item, relative):
        if redirected(item):
            if not item.is_symlink():
                raise RuntimeError(f'Unsupported reparse point: {item}')
            h.update(b'L\0' + relative.encode() + b'\0' + os.readlink(item).encode() + b'\0')
        elif item.is_dir():
            h.update(b'D\0' + relative.encode() + b'\0')
            for child in sorted(item.iterdir()):
                visit(child, relative + '/' + child.name)
        elif item.is_file():
            h.update(b'F\0' + relative.encode() + b'\0')
            with item.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
        else:
            raise RuntimeError(f'Unsupported artifact: {item}')
    visit(path, '.')
    return h.hexdigest()


def git(repo, *args, check=True):
    result = subprocess.run(['git', '-C', str(repo), *args], text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def atomic_marker(marker, state):
    safe_path(marker, leaf=True)
    temporary = marker.with_name(marker.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(state, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        if temporary.exists():
            temporary.unlink()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise RuntimeError('Marker paths must be absolute')
    path = Path(value)
    if '..' in path.parts or '.' in path.parts:
        raise RuntimeError('Marker path traversal is forbidden')
    return safe_path(path)


def validate_state(state, repo, patch_hash):
    if not isinstance(state, dict) or state.get('version') != VERSION:
        raise RuntimeError('Unsupported integration marker version')
    if state.get('patch_version') != patch_hash:
        raise RuntimeError('Integration patch changed; restore the original patch before reverting')
    if not re.fullmatch('[0-9a-f]{32}', str(state.get('installation_id', ''))):
        raise RuntimeError('Invalid installation identifier')
    if state.get('patch_status') not in ('none', 'pending', 'applied'):
        raise RuntimeError('Invalid patch ownership status')
    datetime.datetime.fromisoformat(state['timestamp'])
    output = absolute(state['output_root']) if state.get('output_root') else None
    if output and (output == repo or repo in output.parents):
        raise RuntimeError('Output root must be outside the repository')
    if not isinstance(state.get('links'), list):
        raise RuntimeError('Invalid managed artifact list')
    seen = set()
    for entry in state['links']:
        if not isinstance(entry, dict):
            raise RuntimeError('Invalid artifact entry')
        relative = entry.get('relative')
        if relative not in ARTIFACTS or relative in seen or output is None:
            raise RuntimeError('Marker contains an unexpected or duplicate repository path')
        seen.add(relative)
        dest = safe_path(repo / relative)
        if absolute(entry['target']) != output / relative:
            raise RuntimeError('Marker target does not match the allowlisted output artifact')
        expected_stage = dest.with_name('.viposlt-' + state['installation_id'] + '-' + dest.name + '.staging')
        if absolute(entry['stage']) != expected_stage:
            raise RuntimeError('Invalid artifact staging path')
        if entry.get('status') not in ('pending', 'installed') or entry.get('kind') not in ('pending', 'symlink', 'copy'):
            raise RuntimeError('Invalid managed artifact status')
        if not valid_hash(entry.get('digest')):
            raise RuntimeError('Invalid artifact digest')
        if entry.get('backup') is not None:
            expected = output / 'integration-backups' / state['installation_id'] / relative
            if absolute(entry['backup']) != expected or not valid_hash(entry.get('backup_digest')):
                raise RuntimeError('Invalid backup path or digest')
            safe_path(expected)
        elif entry.get('backup_digest') is not None:
            raise RuntimeError('Unexpected backup digest')
    return state


def owned(path, entry):
    if not exists(path):
        return False
    if path.is_symlink():
        return Path(os.path.abspath(path.parent / os.readlink(path))) == Path(entry['target'])
    if redirected(path):
        return False
    return digest(path) == entry['digest']


def remove_owned(path):
    safe_path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir() and not redirected(path):
        shutil.rmtree(path)
    elif exists(path):
        raise RuntimeError(f'Refusing unsupported managed object: {path}')


def patch_present(repo, patch):
    if git(repo, 'apply', '--reverse', '--check', str(patch), check=False).returncode == 0:
        return True
    if git(repo, 'apply', '--check', str(patch), check=False).returncode == 0:
        return False
    raise RuntimeError('Bridge files have conflicting edits; preserve or resolve them before continuing')


def revert_plan(repo, state, patch):
    reverse = state['patch_status'] != 'none' and patch_present(repo, patch)
    actions = []
    for entry in reversed(state['links']):
        dest, stage = repo / entry['relative'], Path(entry['stage'])
        backup = Path(entry['backup']) if entry['backup'] else None
        backup_exists = backup is not None and exists(backup)
        if backup_exists and digest(backup) != entry['backup_digest']:
            raise RuntimeError(f'Backup was modified; preserve it before reverting: {backup}')
        original_in_place = (backup is not None and not backup_exists and exists(dest)
                             and digest(dest) == entry['backup_digest'])
        for item in (stage, dest):
            if exists(item) and not (item == dest and original_in_place) and not owned(item, entry):
                raise RuntimeError(f'Managed artifact was modified; preserve it before reverting: {item}')
        if backup is not None and not backup_exists and not original_in_place:
            raise RuntimeError(f'Backup is missing: {backup}; refusing to discard the managed artifact')
        actions.append((entry, dest, stage, backup if backup_exists else None, original_in_place))
    return reverse, actions


def revert(repo, marker, state, patch, dry_run):
    reverse, actions = revert_plan(repo, state, patch)
    if reverse:
        print('RUN git -C', repo, 'apply --reverse', patch)
    for entry, dest, stage, backup, original_in_place in actions:
        if exists(stage):
            print('REMOVE managed stage', stage)
        if exists(dest) and not original_in_place:
            print('REMOVE managed artifact', dest)
        if backup:
            print('RESTORE backup', backup, '->', dest)
    print('REMOVE marker', marker)
    if dry_run:
        return 0
    if reverse:
        git(repo, 'apply', '--reverse', str(patch))
    state['patch_status'] = 'none'
    atomic_marker(marker, state)
    for entry, dest, stage, backup, original_in_place in actions:
        if exists(stage):
            remove_owned(stage)
        if exists(dest) and not original_in_place:
            remove_owned(dest)
        if backup:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, dest)
        state['links'].remove(entry)
        atomic_marker(marker, state)
    marker.unlink()
    return 0


def install_artifact(entry, repo, state, marker):
    dest, source, stage = repo / entry['relative'], Path(entry['target']), Path(entry['stage'])
    backup = Path(entry['backup']) if entry['backup'] else None
    moved_backup = False
    created_stage = False
    published = False
    state['links'].append(entry)
    atomic_marker(marker, state)  # Journal intent before moving any user file.
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if backup:
            backup.parent.mkdir(parents=True, exist_ok=True)
            if exists(backup):
                raise RuntimeError(f'Backup already exists; refusing overwrite: {backup}')
            os.replace(dest, backup)
            moved_backup = True
        try:
            stage.symlink_to(source, target_is_directory=source.is_dir())
            created_stage = True
            entry['kind'] = 'symlink'
        except OSError:
            if exists(stage):
                raise RuntimeError(f'Unexpected existing staging object: {stage}')
            created_stage = True
            entry['kind'] = 'copy'
            if source.is_dir():
                shutil.copytree(source, stage, symlinks=True)
            else:
                shutil.copy2(source, stage)
        if not owned(stage, entry):
            raise RuntimeError(f'Prepared artifact changed during installation: {source}')
        if exists(dest):
            raise RuntimeError(f'Destination appeared during installation: {dest}')
        os.replace(stage, dest)
        published = True
        entry['status'] = 'installed'
        atomic_marker(marker, state)
    except BaseException:
        if published and exists(dest):
            remove_owned(dest)
        if created_stage and exists(stage):
            remove_owned(stage)
        if moved_backup:
            if exists(dest):
                raise RuntimeError(f'Cannot restore backup because destination exists: {dest}; backup: {backup}')
            os.replace(backup, dest)
        state['links'].remove(entry)
        atomic_marker(marker, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default=os.environ.get('VIPOSLT_REPO'))
    parser.add_argument('--output-root', default=os.environ.get('VIPOSLT_FEATURE_ROOT'))
    for flag in ('apply-patch', 'link', 'revert', 'force', 'dry-run'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args(argv)
    if not args.repo_root or not any((args.apply_patch, args.link, args.revert)):
        parser.error('--repo-root and at least one of --apply-patch, --link, --revert are required')
    if args.revert and (args.apply_patch or args.link):
        parser.error('--revert cannot be combined with install actions')
    repo = safe_path(Path(args.repo_root).expanduser(), leaf=True)
    if not (repo / 'main.py').is_file() or Path(git(repo, 'rev-parse', '--show-toplevel').stdout.strip()).resolve() != repo:
        parser.error('--repo-root must be the existing serverViPo-SLT checkout root')
    patch = Path(__file__).resolve().parents[1] / 'patches/viposlt_integration.patch'
    patch_bytes = patch.read_bytes()
    patch_hash = hashlib.sha256(patch_bytes).hexdigest()
    patch_paths = set(re.findall(r'^diff --git a/(\S+) b/\S+$', patch_bytes.decode('utf-8-sig'), re.M))
    if not patch_paths or not patch_paths.issubset(PATCH_FILES):
        raise RuntimeError('Patch contains unexpected repository files')
    for relative in patch_paths:
        safe_path(repo / relative, leaf=True)
    marker = safe_path(repo / MARKER, leaf=True)
    state = (validate_state(json.loads(marker.read_text(encoding='utf-8')), repo, patch_hash) if marker.exists()
             else {'version': VERSION, 'installation_id': uuid.uuid4().hex,
                   'patch_version': patch_hash, 'patch_status': 'none', 'output_root': None,
                   'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'links': []})
    if args.revert:
        if not marker.exists():
            raise RuntimeError('No integration marker; refusing an unmanaged revert')
        return revert(repo, marker, state, patch, args.dry_run)
    if state['patch_status'] == 'pending' or any(item['status'] == 'pending' for item in state['links']):
        raise RuntimeError('An interrupted installation exists; run --revert before retrying')
    # If the patch is already present it is retained as preexisting ownership, so a
    # dirty tree caused only by those edits must not block installation.
    apply = args.apply_patch and not patch_present(repo, patch)
    if apply and git(repo, 'status', '--porcelain').stdout.strip() and not args.force:
        raise RuntimeError('Repository is dirty; review changes and use --force only if intended')
    output = safe_path(Path(args.output_root).expanduser(), leaf=True) if args.output_root else None
    if args.link and output is None:
        parser.error('--link requires --output-root or VIPOSLT_FEATURE_ROOT')
    if output and (output == repo or repo in output.parents):
        parser.error('--output-root must be outside the repository')
    if args.link and state['output_root'] and Path(state['output_root']) != output:
        raise RuntimeError('Installed output root differs; revert before relinking')
    before = copy.deepcopy(state)
    had_marker = marker.exists()
    if args.link:
        state['output_root'] = str(output)
    planned = []
    for relative in ARTIFACTS if args.link else ():
        source, dest = safe_path(output / relative, leaf=True), safe_path(repo / relative)
        if not source.exists():
            raise RuntimeError(f'Missing prepared artifact {source}; run prepare first')
        old = next((entry for entry in state['links'] if entry['relative'] == relative), None)
        if old:
            if not owned(dest, old):
                raise RuntimeError(f'Installed artifact changed: {dest}; revert before relinking')
            continue
        backup = output / 'integration-backups' / state['installation_id'] / relative if exists(dest) else None
        if backup and not args.force:
            raise RuntimeError(f'Existing unmanaged artifact: {dest}; use --force to back it up')
        if backup:
            safe_path(backup, leaf=True)
            if exists(backup):
                raise RuntimeError(f'Backup already exists; refusing overwrite: {backup}')
        stage = dest.with_name('.viposlt-' + state['installation_id'] + '-' + dest.name + '.staging')
        if exists(stage):
            raise RuntimeError(f'Staging path already exists: {stage}')
        planned.append({'relative': relative, 'target': str(source), 'digest': digest(source),
                        'kind': 'pending', 'status': 'pending', 'stage': str(stage),
                        'backup': str(backup) if backup else None,
                        'backup_digest': digest(dest) if backup else None})
    if args.apply_patch:
        if apply:
            print('APPLY patch', patch)
            for line in patch_bytes.decode('utf-8-sig').splitlines():
                if line.startswith(('diff --git', '--- ', '+++ ', '@@')):
                    print('  ' + line)
            stat = git(repo, 'apply', '--stat', str(patch), check=False)
            if stat.stdout.strip():
                print(stat.stdout.rstrip())
        else:
            print('Patch already applied; existing ownership retained')
    for entry in planned:
        if entry['backup']:
            print('BACKUP', repo / entry['relative'], '->', entry['backup'])
        print('CREATE symlink (copy fallback)', entry['stage'], '->', entry['target'])
        print('RENAME stage', entry['stage'], '->', repo / entry['relative'])
    print('WRITE atomic marker', marker)
    print('Suggested .gitignore entries (not edited):\n' + MARKER + '\n' + '\n'.join(ARTIFACTS))
    if args.dry_run:
        return 0
    applied = False
    try:
        state['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if apply:
            state['patch_status'] = 'pending'
        atomic_marker(marker, state)
        if apply:
            git(repo, 'apply', str(patch))
            applied = True
            state['patch_status'] = 'applied'
            atomic_marker(marker, state)
        for entry in planned:
            install_artifact(entry, repo, state, marker)
    except BaseException:
        # Roll back only this invocation. Existing managed work keeps its original marker.
        added = [entry for entry in state['links'] if entry['relative'] not in {item['relative'] for item in before['links']}]
        recovery = copy.deepcopy(state)
        recovery['links'] = added
        recovery['patch_status'] = 'applied' if applied else 'none'
        try:
            revert(repo, marker, recovery, patch, False)
            if had_marker:
                atomic_marker(marker, before)
        except Exception as recovery_error:
            atomic_marker(marker, state)
            raise RuntimeError(f'Installation failed and rollback needs attention: {recovery_error}; marker retained') from recovery_error
        raise
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f'Installation failed: {exc}', file=sys.stderr)
        sys.exit(1)
