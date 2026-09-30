"""Apply only the exact, reviewed public-repository repair trees; never merge."""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import urllib.request
import zlib

PAYLOAD_COMMIT = 'ee210d1647d05c209fbeb8341ceae7fbc9d0c6dc'
PAYLOAD_SHA256 = 'c137e89ca7df9e36335b3e9eefb1ad9b7a0728defb399b2048832ad092cbf2c1'
PREFIX = 'https://raw.githubusercontent.com/hyouo/longeclaw/' + PAYLOAD_COMMIT + '/.github/maintenance/'

def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def main():
    parts = []
    for number in (1, 2):
        with urllib.request.urlopen(PREFIX + 'public-repairs.b64.' + str(number), timeout=30) as response:
            parts.append(response.read().decode('ascii'))
    # These three insertion corrections were independently checked against the
    # original local bytes in artifact 11084106413. The digest remains decisive.
    for before, after in [('nuzZZ962ufN', 'nuzZ962ufN'), ('GtscKHv9', 'GtsKHv9'), ('gioKuKuDU', 'gioKuDU')]:
        if sum(p.count(before) for p in parts) != 1:
            raise RuntimeError('Unexpected transport bytes')
        parts = [p.replace(before, after) for p in parts]
    compressed = base64.b64decode(''.join(parts), validate=True)
    if hashlib.sha256(compressed).hexdigest() != PAYLOAD_SHA256:
        raise RuntimeError('Repair payload digest mismatch')
    payload = json.loads(zlib.decompress(compressed))
    repository = os.environ['GITHUB_REPOSITORY']
    owner, name = repository.split('/')
    if owner != 'hyouo' or name not in payload['repositories']:
        raise RuntimeError('Repository is outside the repair allowlist')
    record = payload['repositories'][name]
    if git('rev-parse', 'HEAD') != record['base_sha']:
        raise RuntimeError('Checkout must match the reviewed baseline commit')
    if git('status', '--porcelain'):
        raise RuntimeError('Refusing a dirty checkout')
    root = Path.cwd().resolve()
    for filename, spec in record['files'].items():
        relative = PurePosixPath(filename)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] == '.git':
            raise RuntimeError('Unsafe repair path')
        target = root.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(root) or target.is_symlink():
            raise RuntimeError('Refusing a symlink or escaping repair path')
        prior = spec['before_sha256']
        if prior is None:
            if target.exists():
                raise RuntimeError('New repair file already exists: ' + filename)
            original = b''
        else:
            original = target.read_bytes()
            if hashlib.sha256(original).hexdigest() != prior:
                raise RuntimeError('Baseline bytes changed: ' + filename)
        if spec.get('deleted'):
            target.unlink()
            continue
        updated = original
        for start, count, replacement in reversed(spec['edits']):
            if start < 0 or count < 0 or start + count > len(original):
                raise RuntimeError('Invalid byte edit')
            updated = updated[:start] + replacement.encode('utf-8') + updated[start + count:]
        if hashlib.sha256(updated).hexdigest() != spec['sha256']:
            raise RuntimeError('Repaired file digest mismatch: ' + filename)
        blob = hashlib.sha1(b'blob ' + str(len(updated)).encode() + b'\0' + updated).hexdigest()
        if blob != spec['sha']:
            raise RuntimeError('Repaired Git blob mismatch: ' + filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(updated)
        target.chmod(0o755 if spec['mode'] == '100755' else 0o644)
    git('add', '-A')
    git('diff', '--cached', '--check')
    actual = git('write-tree')
    if actual != record['expected_tree']:
        raise RuntimeError('Full repaired tree mismatch: ' + actual)
    print(repository, 'verified tree', actual, flush=True)
    temporary = Path(os.environ['RUNNER_TEMP'])
    (temporary / 'public-repair-tests.sh').write_text('set -euo pipefail\n' + record['tests'] + '\n')
    (temporary / 'public-repair-record.json').write_text(json.dumps(record, indent=2))
    if '--publish' not in sys.argv:
        return
    branch = record['branch']
    if branch != 'fix/public-repair-20260930':
        raise RuntimeError('Unexpected destination branch')
    current = git('ls-remote', '--heads', 'origin', 'refs/heads/' + record['default_branch']).split()[0]
    if current != record['base_sha']:
        raise RuntimeError('Default branch advanced; review new changes before publishing')
    if git('ls-remote', '--heads', 'origin', 'refs/heads/' + branch):
        raise RuntimeError('Repair branch already exists; do not overwrite it')
    git('config', 'user.name', 'github-actions[bot]')
    git('config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    git('commit', '-m', record['message'])
    git('push', 'origin', 'HEAD:refs/heads/' + branch)
    sha = git('rev-parse', 'HEAD')
    print('PUBLISHED', repository, branch, sha, actual, flush=True)
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
        summary.write(f'Published review branch `{branch}` at `{sha}` with verified tree `{actual}`. No merge.\n')


if __name__ == '__main__':
    main()
