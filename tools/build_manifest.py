"""Deliberately refresh hashes after reviewing changes to saved evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    *ROOT.glob('supplemental_*.json'), *ROOT.glob('fig*.png'), *ROOT.glob('fig*.pdf'),
    *ROOT.glob('paper/**/*.pdf'), *ROOT.glob('archive/*'),
    *ROOT.glob('output/**/*.json'), *ROOT.glob('output/**/*.npz'),
    *ROOT.glob('output/**/*.txt'), *ROOT.glob('output/**/*.csv'),
    *ROOT.glob('output/**/*.png'), *ROOT.glob('output/**/*.pdf'),
]

entries = []
for path in sorted(set(FILES)):
    data = path.read_bytes()
    entries.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()})
(ROOT / 'verification/artifact_manifest.json').write_text(
    json.dumps({'algorithm': 'SHA-256', 'files': entries}, indent=2) + '\n', encoding='utf-8')
print(f'Wrote {len(entries)} artifact checksums. Review the diff before committing.')
