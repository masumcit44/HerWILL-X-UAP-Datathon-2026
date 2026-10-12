"""Rebuild Submission 3's ID-derived SID source (config A) on fold set B, verbatim from
submission_reproducible_file_3.ipynb cells 11-12. Output: sid.npz with oof, test, fold_ids, meta(uses_id=true).
This source is ID-DERIVED. It is used only by the performance track and the disclosed ablation.

Usage: python build_sid_source.py --data-dir . --folds final_sources/fold_ids.npy --out cache/v7_text/sid.npz
Expected OOF: 0.627895 (sklearn 1.9) / 0.627972 (sklearn 1.8); max_features tie-breaking differs by version.
"""
import argparse, hashlib, json
import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="."); ap.add_argument("--folds",default="final_sources/fold_ids.npy"); ap.add_argument("--out",default="cache/v7_text/sid.npz")
a=ap.parse_args()
d=a.data_dir.rstrip("/")+"/"
tr=pd.read_csv(d+'train.csv'); te=pd.read_csv(d+'test.csv'); folds=np.load(a.folds).astype(np.int64)
assert hashlib.sha256(folds.tobytes()).hexdigest()=="cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca", "not fold set B"
URL=re.compile(r"(?:https?://|www\.)\S+",re.I); M=re.compile(r"@\w+"); W=re.compile(r"\s+", re.UNICODE)
cl=lambda s: W.sub(" ",M.sub(" USER ",URL.sub(" URL ",str(s)))).strip()
ctr=tr.text.astype(str).map(cl).to_numpy(); cte=te.text.astype(str).map(cl).to_numpy()
ids=tr.id.to_numpy(np.int64); tids=te.id.to_numpy(np.int64); y=tr.y.to_numpy(int)
V=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),min_df=2,sublinear_tf=True,max_features=80000,dtype=np.float32).fit(ctr)
Xn=normalize(V.transform(ctr)).tocsr()
def sid_prior(q_ids,Xq,pool_rows,ids,y,Xpool,KW=120,k=180,beta=10.0,gamma=1.0,chunk=800):
    o=np.argsort(ids[pool_rows],kind="stable"); pid=ids[pool_rows][o]; prow=pool_rows[o]
    out=np.zeros((len(q_ids),3))
    for s in range(0,len(q_ids),chunk):
        qi=q_ids[s:s+chunk]; xq=Xq[s:s+chunk]; pos=np.searchsorted(pid,qi)
        cand=pos[:,None]+np.arange(-KW,KW)[None,:]; valid=(cand>=0)&(cand<len(pid)); cc=np.clip(cand,0,len(pid)-1)
        d=np.abs(pid[cc]-qi[:,None]).astype(float); d[~valid]=np.inf
        order=np.argsort(d,axis=1,kind="stable")[:,:k]; dsel=np.take_along_axis(d,order,1); nrow=prow[np.take_along_axis(cc,order,1)]
        qrep=np.repeat(np.arange(xq.shape[0]),nrow.shape[1])
        sim=np.asarray(xq[qrep].multiply(Xpool[nrow.ravel()]).sum(1)).reshape(nrow.shape)
        w=np.where(np.isfinite(dsel),1.0/((np.where(np.isfinite(dsel),dsel,1.0)+1e-6)**gamma),0.0)*np.exp(beta*sim)
        yy=y[nrow]; P=np.stack([(w*(yy==c)).sum(1) for c in range(3)],1); tot=P.sum(1,keepdims=True)
        out[s:s+chunk]=np.where(tot>0,P/np.where(tot>0,tot,1),1/3)
    return out
oof=np.zeros((len(y),3)); t0=time.time()
for k in range(5):
    va=np.flatnonzero(folds==k); trr=np.flatnonzero(folds!=k); oof[va]=sid_prior(ids[va],Xn[va],trr,ids,y,Xn); print(k,round(time.time()-t0),flush=True)
tst=sid_prior(tids,normalize(V.transform(cte)).tocsr(),np.arange(len(y)),ids,y,Xn)
score=f1_score(y,oof.argmax(1),average='macro')
np.savez(a.out,oof=oof,test=tst,fold_ids=folds,meta=np.array(json.dumps(dict(name='sid_config_A',uses_id=True,oof_macro_f1=score))))
print('SID OOF',score,'->',a.out)
