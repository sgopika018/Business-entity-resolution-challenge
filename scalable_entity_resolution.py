#!/usr/bin/env python3
"""Scalable entity resolution for the multi-million-row challenge dataset.

This version avoids loading all TSVs into RAM. It builds an SQLite index over
Source 2 and Source 3, then streams Source 1 and writes the two required TSVs.
"""
from __future__ import annotations
import argparse, csv, re, sqlite3, sys, time
from pathlib import Path
from collections import defaultdict

LEGAL={"limited":"ltd","ltd":"","private":"pvt","pvt":"","priv":"","corporation":"corp","company":"co","incorporated":"inc"}
ABBR={"street":"st","road":"rd","avenue":"ave","apartment":"apt","building":"bldg","boulevard":"blvd","highway":"hwy","place":"pl","lane":"ln","drive":"dr","suite":"ste"}

def norm(x):
    s=(x or "").lower().replace("&"," and ")
    s=re.sub(r"[^\w\s]"," ",s,flags=re.UNICODE)
    out=[]
    for t in s.split():
        t=LEGAL.get(t,t); t=ABBR.get(t,t)
        if t: out.append(t)
    return " ".join(out)
def compact(s): return re.sub(r"[^a-z0-9]","",s)
def grams(s,n=3): return {s[i:i+n] for i in range(max(0,len(s)-n+1))}
def jacc(a,b):
    A=set(a.split()); B=set(b.split()); return len(A&B)/len(A|B) if A|B else 0.0
def containment(a,b):
    A=set(a.split()); B=set(b.split()); return max(len(A&B)/len(A),len(A&B)/len(B)) if A and B else 0.0
def char_jacc(a,b):
    A=grams(a); B=grams(b); return len(A&B)/len(A|B) if A|B else 0.0
def sim(a,b):
    ns1,ns2,as1,as2=a[1],b[1],a[2],b[2]
    nc1,nc2,ac1,ac2=compact(ns1),compact(ns2),compact(as1),compact(as2)
    name=max(jacc(ns1,ns2), char_jacc(nc1,nc2))
    addr=max(jacc(as1,as2), char_jacc(ac1,ac2))
    exact_name=bool(nc1 and nc1==nc2); exact_addr=bool(ac1 and ac1==ac2)
    country=float(a[3]==b[3] and bool(a[3]))
    # Exact field agreement is highly precision-oriented; partial similarity is supportive.
    return (0.43*float(exact_name)+0.34*float(exact_addr)+0.13*name+0.08*addr+0.02*country, exact_name, exact_addr)

def rows(path):
    with open(path,encoding='utf-8',newline='',errors='replace') as f:
        r=csv.DictReader(f,delimiter='\t')
        for x in r: yield x

def build_db(dbpath,s2,s3):
    if dbpath.exists(): dbpath.unlink()
    con=sqlite3.connect(dbpath); con.execute('PRAGMA journal_mode=WAL'); con.execute('PRAGMA synchronous=OFF'); con.execute('PRAGMA temp_store=FILE')
    con.execute('CREATE TABLE rec (id TEXT PRIMARY KEY, name TEXT, addr TEXT, country TEXT, nk TEXT, ak TEXT, np TEXT, az TEXT)')
    con.execute('CREATE INDEX ix_nk ON rec(nk)'); con.execute('CREATE INDEX ix_ak ON rec(ak)'); con.execute('CREATE INDEX ix_np ON rec(np)'); con.execute('CREATE INDEX ix_az ON rec(az)')
    batch=[]; count=0
    for path in (s2,s3):
        for x in rows(path):
            n=norm(x.get('business_name','')); a=norm(x.get('business_address','')); c=norm(x.get('country',''))
            nc,ac=compact(n),compact(a)
            batch.append((x['entity_id'],n,a,c,nc,ac,(c+' '+nc[:10]).strip() if nc else '',(c+' '+ac[-10:]).strip() if ac else ''))
            if len(batch)>=10000:
                con.executemany('INSERT INTO rec VALUES (?,?,?,?,?,?,?,?)',batch); count+=len(batch); batch.clear()
                if count%500000==0: print(f'indexed {count:,}',flush=True)
    if batch: con.executemany('INSERT INTO rec VALUES (?,?,?,?,?,?,?,?)',batch); count+=len(batch)
    con.commit(); con.execute('VACUUM'); con.close(); print(f'index complete: {count:,} records',flush=True)

def get_candidates(con,n,a,c):
    nc,ac=compact(n),compact(a); np=(c+' '+nc[:10]).strip() if nc else ''; az=(c+' '+ac[-10:]).strip() if ac else ''
    ids=set();
    # Exact keys are primary blocks. Limits prevent pathological common-key expansion.
    for col,key in [('nk',nc),('ak',ac),('np',np),('az',az)]:
        if not key: continue
        q=f'SELECT id FROM rec WHERE {col}=? LIMIT 500'
        ids.update(x[0] for x in con.execute(q,(key,)))
    return ids

def write_header(path,cols):
    path.parent.mkdir(parents=True,exist_ok=True); f=open(path,'w',encoding='utf-8',newline=''); csv.writer(f,delimiter='\t',lineterminator='\n').writerow(cols); return f

def predict(data,out,dbfile,threshold):
    test=data/'dataset/test'; s1=test/'test_source1.tsv'; s2=test/'test_source2.tsv'; s3=test/'test_source3.tsv'
    if not dbfile.exists(): build_db(dbfile,s2,s3)
    con=sqlite3.connect(dbfile); con.execute('PRAGMA cache_size=-262144')
    mf=write_header(out/'matching_results.tsv',['source1_entity_id','matched_entity_ids']); cf=write_header(out/'candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids'])
    mw,cw=csv.writer(mf,delimiter='\t',lineterminator='\n'),csv.writer(cf,delimiter='\t',lineterminator='\n')
    t=time.time(); total=0
    for x in rows(s1):
        eid=x['entity_id']; n=norm(x.get('business_name','')); a=norm(x.get('business_address','')); c=norm(x.get('country',''))
        ids=get_candidates(con,n,a,c); records=[]
        if ids:
            qmarks=','.join('?'*len(ids)); records=list(con.execute(f'SELECT id,name,addr,country FROM rec WHERE id IN ({qmarks})',tuple(ids)))
        base=(eid,n,a,c); scored=[]
        for r in records:
            score,en,ea=sim(base,r)
            if score>=threshold: scored.append((score,r[0],en,ea))
        # Keep all strong exact candidates; preserve deterministic ID ordering.
        matched=sorted({r[1] for r in scored})
        cand=sorted(ids)
        mw.writerow([eid,','.join(matched)]); cw.writerow([eid,','.join(cand)])
        total+=1
        if total%100000==0: print(f'processed {total:,} Source 1 rows in {(time.time()-t)/60:.1f} min',flush=True)
    mf.close(); cf.close(); con.close(); print(f'outputs written for {total:,} Source 1 rows',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='.'); p.add_argument('--mode',choices=['predict','build-index'],default='predict'); p.add_argument('--threshold',type=float,default=.62); p.add_argument('--index',default='entity_index.sqlite'); args=p.parse_args()
    root=Path(args.data_dir); train=root/'dataset/train'; test=root/'dataset/test'; db=Path(args.index)
    if not db.is_absolute(): db=root/db
    if args.mode=='build-index': build_db(db,test/'test_source2.tsv',test/'test_source3.tsv')
    else: predict(root,root/'output',db,args.threshold)
if __name__=='__main__': main()
