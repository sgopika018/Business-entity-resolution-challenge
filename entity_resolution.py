#!/usr/bin/env python3
"""Business entity resolution pipeline for the Unstop/Amazon ML challenge.

Usage:
  python src/entity_resolution.py --data-dir . --mode predict
  python src/entity_resolution.py --data-dir . --mode validate
  python src/entity_resolution.py --data-dir . --mode train-evaluate
"""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support

ID_RE = re.compile(r"[^a-z0-9]+")
LEGAL = {"limited":"ltd", "ltd":"", "private":"pvt", "pvt":"", "priv":"", "corporation":"corp", "company":"co", "incorporated":"inc"}
ABBR = {"street":"st", "road":"rd", "avenue":"ave", "apartment":"apt", "building":"bldg", "boulevard":"blvd", "highway":"hwy", "place":"pl", "lane":"ln", "drive":"dr", "suite":"ste", "near":"near"}

def norm_text(x) -> str:
    if pd.isna(x): return ""
    s = str(x).lower().replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    toks = []
    for t in s.split():
        t = LEGAL.get(t, t)
        t = ABBR.get(t, t)
        if t: toks.append(t)
    return " ".join(toks)

def compact(s): return re.sub(r"[^a-z0-9]", "", s)
def token_set(s): return set(s.split()) if s else set()
def jaccard(a,b):
    A, B = token_set(a), token_set(b)
    return len(A&B)/len(A|B) if A|B else 0.0
def containment(a,b):
    A, B = token_set(a), token_set(b)
    return max(len(A&B)/len(A), len(A&B)/len(B)) if A and B else 0.0
def dice(a,b):
    A, B = token_set(a), token_set(b)
    return 2*len(A&B)/(len(A)+len(B)) if A and B else 0.0
def char_jaccard(a,b,n=3):
    A={a[i:i+n] for i in range(max(0,len(a)-n+1))}; B={b[i:i+n] for i in range(max(0,len(b)-n+1))}
    return len(A&B)/len(A|B) if A|B else 0.0

def load(path):
    df=pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).fillna("")
    required={"entity_id","business_name","business_address","country"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for c in ["business_name","business_address","country"]: df[c+"_norm"]=df[c].map(norm_text)
    df["name_compact"]=df.business_name_norm.map(compact); df["address_compact"]=df.business_address_norm.map(compact)
    return df

def read_truth(path):
    gt=pd.read_csv(path,sep="\t",dtype=str,keep_default_na=False).fillna("")
    out={}
    for _,r in gt.iterrows(): out[r.source1_entity_id]=set(x for x in r.matched_entity_ids.split(",") if x)
    return out

def features(a,b):
    n1,n2=a.business_name_norm,b.business_name_norm; ad1,ad2=a.business_address_norm,b.business_address_norm
    return np.array([jaccard(n1,n2), containment(n1,n2), dice(n1,n2), char_jaccard(a.name_compact,b.name_compact),
                     jaccard(ad1,ad2), containment(ad1,ad2), dice(ad1,ad2), char_jaccard(a.address_compact,b.address_compact),
                     float(a.country_norm==b.country_norm), float(bool(a.name_compact and a.name_compact==b.name_compact)),
                     float(bool(a.address_compact and a.address_compact==b.address_compact))], dtype=float)

def candidates(s1, pool, max_neighbors=80):
    """High-recall blocking: exact keys plus TF-IDF character nearest neighbors."""
    n=len(pool); result=defaultdict(set)
    keys=defaultdict(set)
    for j,r in pool.iterrows():
        for key in [r.name_compact, r.address_compact, r.country_norm+" "+r.name_compact[:8], r.country_norm+" "+r.address_compact[-8:]]:
            if key: keys[key].add(j)
    texts=(pool.business_name_norm+" "+pool.business_address_norm).tolist()
    qtexts=(s1.business_name_norm+" "+s1.business_address_norm).tolist()
    alltexts=texts+qtexts
    if n:
        vec=TfidfVectorizer(analyzer="char",ngram_range=(2,5),min_df=1,sublinear_tf=True)
        X=vec.fit_transform(alltexts)
        nn=NearestNeighbors(n_neighbors=min(max_neighbors,n),metric="cosine",algorithm="brute").fit(X[:n])
        _, inds=nn.kneighbors(X[n:])
    else: inds=[]
    for qi,(_,r) in enumerate(s1.iterrows()):
        for key in [r.name_compact, r.address_compact, r.country_norm+" "+r.name_compact[:8], r.country_norm+" "+r.address_compact[-8:]]:
            result[r.entity_id].update(pool.iloc[list(keys.get(key,[]))].entity_id.tolist())
        if n: result[r.entity_id].update(pool.iloc[inds[qi]].entity_id.tolist())
    return result

def build_model(train1, pool, truth, cand):
    X=[]; y=[]
    idmap=pool.set_index("entity_id")
    rng=np.random.default_rng(42)
    for _,a in train1.iterrows():
        positives=truth.get(a.entity_id,set())
        cands=list(cand.get(a.entity_id,set()))
        for bid in cands:
            if bid not in idmap.index: continue
            X.append(features(a,idmap.loc[bid])); y.append(int(bid in positives))
        neg=[x for x in cands if x not in positives]
        # Keep additional hard negatives; blocking candidates are already difficult cases.
        if len(neg)>120: neg=rng.choice(neg,120,replace=False)
    if len(set(y))<2: return None
    model=LogisticRegression(max_iter=1000,class_weight="balanced",C=2.0)
    model.fit(np.asarray(X),np.asarray(y)); return model

def score_pair(a,b,model):
    f=features(a,b)
    if model is not None: return float(model.predict_proba(f.reshape(1,-1))[0,1])
    # conservative fallback when the training labels are too small
    return float(0.35*f[0]+0.35*f[4]+0.15*f[3]+0.15*f[7])

def infer(s1,pool,cand,model,threshold=0.62):
    idmap=pool.set_index("entity_id"); matches={}
    for _,a in s1.iterrows():
        scored=[]
        for bid in cand.get(a.entity_id,set()):
            if bid in idmap.index: scored.append((score_pair(a,idmap.loc[bid],model),bid))
        scored.sort(reverse=True)
        chosen=[]
        for score,bid in scored:
            # A second guard reduces false merges, important for macro F0.5.
            if score>=threshold: chosen.append(bid)
        matches[a.entity_id]=sorted(set(chosen))
    return matches

def write_outputs(s1,cand,matches,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    with open(outdir/"candidate_pairs.tsv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,delimiter="\t",lineterminator="\n"); w.writerow(["source1_entity_id","candidate_entity_ids"])
        for eid in s1.entity_id: w.writerow([eid,",".join(sorted(cand.get(eid,set())))])
    with open(outdir/"matching_results.tsv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,delimiter="\t",lineterminator="\n"); w.writerow(["source1_entity_id","matched_entity_ids"])
        for eid in s1.entity_id: w.writerow([eid,",".join(matches.get(eid,[]))])

def evaluate(s1,pool,truth,model,thresholds=(.45,.50,.55,.60,.65,.70,.75)):
    cand=candidates(s1,pool); print("threshold precision recall f0.5")
    for t in thresholds:
        pred=infer(s1,pool,cand,model,t); vals=[]
        for eid in s1.entity_id:
            p=pred[eid]; y=truth.get(eid,set()); tp=len(p&y)
            if not p and not y: vals.append(1.0); continue
            prec=tp/len(p) if p else 0; rec=tp/len(y) if y else 0
            vals.append((1.25*prec*rec/(.25*prec+rec)) if (.25*prec+rec) else 0)
        print(f"{t:.2f} {np.mean(vals):.6f}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="."); ap.add_argument("--mode",choices=["predict","train-evaluate","validate"],default="predict"); ap.add_argument("--threshold",type=float,default=.62); args=ap.parse_args()
    root=Path(args.data_dir); train=root/"dataset/train"; test=root/"dataset/test"; out=root/"output"
    if args.mode=="validate":
        from validate_submission import validate
        raise SystemExit(validate(test,out))
    files=[train/"train_source1.tsv",train/"train_source2.tsv",train/"train_source3.tsv",train/"train_ground_truth.tsv",test/"test_source1.tsv",test/"test_source2.tsv",test/"test_source3.tsv"]
    missing=[str(x) for x in files if not x.exists()]
    if missing: raise FileNotFoundError("Missing required files:\n"+"\n".join(missing))
    tr1=load(files[0]); pooltr=pd.concat([load(files[1]),load(files[2])],ignore_index=True); truth=read_truth(files[3])
    trcand=candidates(tr1,pooltr); model=build_model(tr1,pooltr,truth,trcand)
    if args.mode=="train-evaluate": evaluate(tr1,pooltr,truth,model); return
    te1=load(files[4]); poolte=pd.concat([load(files[5]),load(files[6])],ignore_index=True)
    tecand=candidates(te1,poolte); matches=infer(te1,poolte,tecand,model,args.threshold); write_outputs(te1,tecand,matches,out)
    print(f"Wrote {out/'matching_results.tsv'} and {out/'candidate_pairs.tsv'}")

if __name__=="__main__": main()
