"""Import explicitly selected local materials as immutable extracted snapshots."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from .storage import atomic_json


async def extract_material(path, timeout=30):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Material is not an existing regular file: {path}')
    proc = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).with_name('material_worker.py')),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(json.dumps({'path': str(path)}).encode()), timeout)
        except TimeoutError:
            raise ValueError(f'Material extraction timed out after {timeout}s: {path.name}') from None
        if proc.returncode:
            raise ValueError(f'Material extraction failed: {path.name}')
        result = json.loads(stdout)
        if 'error' in result:
            raise ValueError(f'{path.name}: {result["error"]}')
        return result
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


async def import_materials(store, journal, proposals=(), materials=(), timeout=30):
    entries = [('proposal', Path(p)) for p in proposals] + [('reference', Path(p)) for p in materials]
    if len(entries) > 50:
        raise ValueError('Import at most 50 files per command')
    # Extract the entire batch before changing evidence: an invalid file cannot leave a partial batch.
    prepared = []
    for role, path in entries:
        journal.event('materials', 'import_start', f'Reading {role}: {path.name}')
        try:
            extracted = await extract_material(path, timeout)
        except Exception as error:
            journal.event('materials', 'import_error', str(error))
            raise
        identity = f'user-material:{role}:{extracted["sha256"]}'
        sid = 'S' + hashlib.sha256(identity.encode()).hexdigest()[:10]
        prepared.append((role, path, identity, sid, extracted))
    imported = []
    for role, path, identity, sid, extracted in prepared:
        existing = store.state['sources'].get(sid)
        directory = store.directory / 'materials'
        directory.mkdir(exist_ok=True, mode=0o700)
        snapshot = f'materials/{sid}.txt'
        text = journal.clean(extracted.pop('text'))
        target = store.directory / snapshot
        if existing and target.is_file() and not target.is_symlink():
            try:
                intact = target.read_text(encoding='utf-8') == text
            except (OSError, UnicodeError):
                intact = False
            if intact:
                journal.event('materials', 'duplicate', f'Already imported {role}: {path.name}', source_id=sid)
                continue
        temp = target.with_suffix('.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(target)
        metadata = {**extracted, 'material_role': role, 'snapshot': snapshot, 'content_chars': len(text),
                    'imported_at': existing['imported_at'] if existing else datetime.now(timezone.utc).isoformat(),
                    'snapshot_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    'evidence_type': 'user_proposal' if role == 'proposal' else 'user_material'}
        actual_id, _ = store.source(path.name, snapshot, text, 'user', paper_id=identity, **metadata)
        assert actual_id == sid
        if not existing:
            imported.append(sid)
        verb = 'Restored' if existing else 'Imported'
        journal.event('materials', 'restored' if existing else 'imported', f'{verb} {role}: {path.name} ({len(text)} characters' +
                      ('; truncated)' if extracted['truncated'] else ')'), source_id=sid, **metadata)
    manifest = [{k: v for k, v in source.items() if k != 'text'} for source in store.state['sources'].values()
                if source.get('provider') == 'user']
    if entries:
        atomic_json(store.directory / 'materials.json', manifest)
    return imported
