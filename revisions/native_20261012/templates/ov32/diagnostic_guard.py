"""Process-local fail-closed write scope; never deletes any server data."""
import os
from pathlib import Path
import sys
import tempfile
import re


def fresh_contract_rename(root, args):
    """Only the single-writer Journal's fresh contract commit, never replacement."""
    source, destination, source_fd, destination_fd = args
    if source_fd not in (-1, None) or destination_fd not in (-1, None):
        return False
    source, destination = Path(source).resolve(), Path(destination).resolve()
    return (root in source.parents and source.parent == destination.parent
            and re.fullmatch(r'contract\.[0-9a-f]{32}\.pending', source.name) is not None
            and destination.name == 'contract.json' and source.is_file()
            and not os.path.lexists(destination))


def install(run_root):
    root = Path(run_root).resolve(strict=True)
    temporary = root / 'tmp'
    temporary.mkdir(exist_ok=True)
    # Avoid tempfile's gettempdir probe, which ordinarily creates and unlinks.
    tempfile.tempdir = str(temporary)
    os.environ['TMPDIR'] = str(temporary)
    sys.dont_write_bytecode = True
    def audit(event, args):
        if event == 'os.rename' and fresh_contract_rename(root, args):
            return
        if event in {'os.remove', 'os.rmdir', 'os.rename', 'os.truncate', 'os.link', 'os.symlink'}:
            raise PermissionError('Server no-delete guard blocked ' + event)
        if event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'}:
            raise PermissionError('No child processes in GPU gate: ' + event)
        if event == 'open':
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if not writing or isinstance(path, int): return
            if os.fspath(path) == '/dev/null': return
            resolved = Path(os.fsdecode(path)).resolve()
            if resolved != root and root not in resolved.parents:
                raise PermissionError('Write outside new run directory: ' + str(resolved))
        if event == 'os.mkdir':
            path = Path(os.fsdecode(args[0])).resolve()
            if path != root and root not in path.parents:
                raise PermissionError('mkdir outside new run directory: ' + str(path))
    sys.addaudithook(audit)
    sys._stage_no_delete_guard = True
    return root
