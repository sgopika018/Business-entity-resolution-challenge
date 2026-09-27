#!/usr/bin/env python3
import csv, sys
from pathlib import Path

def read_ids(path, expected_cols):
    with open(path,encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    if not rows: return [], []
    errors=[]
    if list(rows[0].keys()) != expected_cols: errors.append(f'{path}: columns must be {expected_cols}')
    return rows, errors

def validate(test_dir, out_dir):
    test_dir=Path(test_dir); out_dir=Path(out_dir); errors=[]
    s1_path=test_dir/'test_source1.tsv'; s2_path=test_dir/'test_source2.tsv'; s3_path=test_dir/'test_source3.tsv'
    try:
        def ids(p):
            with open(p,encoding='utf-8',newline='') as f: return {r['entity_id'] for r in csv.DictReader(f,delimiter='\t')}
        s1,s2,s3=ids(s1_path),ids(s2_path),ids(s3_path)
    except Exception as e: return print(f'ERROR reading test files: {e}') or 1
    valid=s2|s3
    for name, key in [('matching_results.tsv','matched_entity_ids'),('candidate_pairs.tsv','candidate_entity_ids')]:
        p=out_dir/name
        if not p.exists(): errors.append(f'missing {p}'); continue
        rows,es=read_ids(p,['source1_entity_id',key]); errors+=es
        seen=set()
        for i,r in enumerate(rows,2):
            sid=r.get('source1_entity_id',''); vals=[x for x in r.get(key,'').split(',') if x]
            if sid in seen: errors.append(f'{name}:{i}: duplicate source1_entity_id {sid}')
            seen.add(sid)
            if sid not in s1: errors.append(f'{name}:{i}: unknown source1_entity_id {sid}')
            if len(vals)!=len(set(vals)): errors.append(f'{name}:{i}: duplicate IDs')
            bad=[x for x in vals if x not in valid]
            if bad: errors.append(f'{name}:{i}: invalid candidate/match IDs {bad[:3]}')
        if seen != s1: errors.append(f'{name}: must contain every test Source 1 ID exactly once; missing {len(s1-seen)}')
    mp=out_dir/'matching_results.tsv'; cp=out_dir/'candidate_pairs.tsv'
    if mp.exists() and cp.exists():
        mrows,_=read_ids(mp,['source1_entity_id','matched_entity_ids']); crows,_=read_ids(cp,['source1_entity_id','candidate_entity_ids'])
        cm={r['source1_entity_id']:set(x for x in r['candidate_entity_ids'].split(',') if x) for r in crows}
        for r in mrows:
            bad=set(x for x in r['matched_entity_ids'].split(',') if x)-cm.get(r['source1_entity_id'],set())
            if bad: errors.append(f"matching result for {r['source1_entity_id']} is not a subset of candidates")
    if errors:
        print('FAIL'); print('\n'.join(f'{i}. {e}' for i,e in enumerate(errors,1))); return 1
    print('PASS'); return 0

if __name__=='__main__':
    if len(sys.argv)==3: raise SystemExit(validate(sys.argv[1],sys.argv[2]))
    print('Usage: python src/validate_submission.py dataset/test output'); raise SystemExit(2)
