#!/usr/bin/env python3
"""Publish a generated training corpus as a versioned source-controlled resource."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',required=True)
    p.add_argument('--source-root',type=Path,default=Path('outputs/train_data'))
    p.add_argument('--destination-root',type=Path,default=Path('src/slm_training/resources/train_data'))
    args=p.parse_args(argv)
    if not args.version.replace('_','').replace('-','').replace('.','').isalnum(): raise SystemExit('invalid version')
    source=args.source_root/args.version; dest=args.destination_root/args.version
    if not (source/'records.jsonl').is_file() or not (source/'manifest.json').is_file(): raise SystemExit(f'incomplete corpus: {source}')
    manifest=json.loads((source/'manifest.json').read_text())
    if not manifest.get('record_count') and not manifest.get('records'): raise SystemExit('manifest has no record count')
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('records.jsonl','manifest.json','stats.json'):
        src=source/name
        if src.is_file(): shutil.copyfile(src,dest/name)
    print(json.dumps({'version':args.version,'destination':str(dest),'record_count':manifest.get('record_count'),'manifest_sha':manifest.get('manifest_sha256')},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
