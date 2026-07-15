#!/usr/bin/env python3
"""Publish reproducible per-step experiment and ladder evidence to docs/design."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def read(path: Path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError): return None

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs',type=Path,default=Path('outputs'))
    p.add_argument('--docs',type=Path,default=Path('docs/design'))
    p.add_argument('--generated-at',default=None)
    a=p.parse_args(); runs=[]
    for path in sorted(a.outputs.glob('runs/*/train_summary.json')):
        summary=read(path)
        if not isinstance(summary,dict): continue
        steps=[]
        for row in summary.get('nll_history') or []:
            if not isinstance(row,dict): continue
            item={k:row.get(k) for k in ('step','weighted_nll','broad_mean_nll','broad_constraint_rescue_gap','seen_target_tokens','complete')}
            loss=read(path.parent/f"loss_suites_step_{row.get('step')}.json")
            if isinstance(loss,dict): item['loss_suites']=loss.get('suites') or loss.get('metrics') or loss
            steps.append(item)
        runs.append({'run_id':summary.get('run_id') or path.parent.name,'recipe':summary.get('recipe') or {},'track':summary.get('track') or {},'accel':summary.get('accel') or {},'telemetry':summary.get('telemetry') or {},'steps':steps,'best_weighted_nll':summary.get('best_weighted_nll'),'data_manifest_sha':summary.get('data_manifest_sha')})
    ladders=[]
    for path in sorted(a.outputs.glob('ladders/**/ladder_summary.json')):
        value=read(path)
        if value is not None: ladders.append(value)
    a.docs.mkdir(parents=True,exist_ok=True)
    stamp=a.generated_at or 'unspecified'
    (a.docs/'experiment-step-data.json').write_text(json.dumps({'schema_version':1,'generated_at':stamp,'source':'outputs/runs/*/train_summary.json and loss_suites_step_*.json','runs':runs},indent=2)+'\n')
    (a.docs/'ladder-step-data.json').write_text(json.dumps({'schema_version':1,'generated_at':stamp,'source':'outputs/ladders/**/ladder_summary.json','ladders':ladders},indent=2)+'\n')
    print(json.dumps({'runs':len(runs),'ladders':len(ladders)}))
    return 0
if __name__=='__main__': raise SystemExit(main())
