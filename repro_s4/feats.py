import numpy as np, pandas as pd, re
W=''; d='./'
tr=pd.read_csv(d+'train.csv'); te=pd.read_csv(d+'test.csv'); folds=np.load('final_sources/fold_ids.npy').astype(np.int64)
y=tr.y.to_numpy(np.int64); ids=tr.id.to_numpy(np.int64); tids=te.id.to_numpy(np.int64)
SRC={'tfidf':'cache/v7_text/text_tfidf_final.npz','knn':'cache/v7_text/text_knn_final.npz','e5':'final_sources/emb_e5-base_final.npz',
     'sid':'cache/v7_text/sid.npz','pid':'cache/v7_text/plain_id.npz','idh':'final_sources/xlmr_base_idhead_final.npz'}
def nb_feats(q, pool_ids, pool_y):
    o=np.argsort(pool_ids); s=pool_ids[o]; sy=pool_y[o]; n=len(s)
    pos=np.searchsorted(s,q); F={}
    for i in (1,2,3):
        p=pos-i; ok=p>=0; pc=np.clip(p,0,n-1)
        F[f'prev{i}_y']=np.where(ok,sy[pc],-1); F[f'prev{i}_d']=np.where(ok,np.log1p(q-s[pc]),9.0)
        nx=pos+i-1; ok=nx<n; nc=np.clip(nx,0,n-1)
        F[f'next{i}_y']=np.where(ok,sy[nc],-1); F[f'next{i}_d']=np.where(ok,np.log1p(s[nc]-q),9.0)
    cum=np.vstack([np.zeros((1,3)),np.cumsum(np.eye(3)[sy],0)])
    for Wn in (3,10,30,100,300):
        a=np.searchsorted(s,q-Wn,'left'); b=np.searchsorted(s,q+Wn,'right'); c=cum[b]-cum[a]; t=c.sum(1)
        F[f'w{Wn}_n']=t
        for k in range(3): F[f'w{Wn}_p{k}']=np.where(t>0,c[:,k]/np.maximum(t,1),np.nan)
    return pd.DataFrame(F)
def text_meta(txt):
    t=pd.Series(txt).astype(str); L=t.str.len()
    return pd.DataFrame({'len':L,'nwords':t.str.split().str.len(),
      'bn':t.str.contains('[ঀ-৿]').astype(int),'lat':t.str.contains('[A-Za-z]').astype(int),
      'emoji':t.str.count('[\U0001F300-\U0001FAFF☀-➿]'),'caps':t.str.count('[A-Z]')/L.clip(lower=1),
      'excl':t.str.count('!'),'q':t.str.count(r'\?'),'digits':t.str.count(r'\d')/L.clip(lower=1)})
# OOF features: neighbour pool = train rows of the other folds
Xtr=[None]*len(y); parts=[]
nbtr=pd.DataFrame(index=range(len(y)))
for k in range(5):
    va=np.flatnonzero(folds==k); pool=np.flatnonzero(folds!=k)
    f=nb_feats(ids[va],ids[pool],y[pool]); f.index=va; parts.append(f)
nbtr=pd.concat(parts).sort_index()
nbte=nb_feats(tids,ids,y)
def probs(split):
    out={}
    for n,p in SRC.items():
        z=np.load(p); a=z['oof' if split=='tr' else 'test']
        for k in range(3): out[f'{n}_p{k}']=a[:,k]
    return pd.DataFrame(out)
Ftr=pd.concat([probs('tr'),nbtr.reset_index(drop=True),text_meta(tr.text)],axis=1); Ftr['rid']=ids
Fte=pd.concat([probs('te'),nbte,text_meta(te.text)],axis=1); Fte['rid']=tids
Ftr.to_parquet('Ftr.parquet'); Fte.to_parquet('Fte.parquet'); print(Ftr.shape,Fte.shape)
print(Ftr[['prev1_d','next1_d','w10_n']].describe().loc[['mean','50%']]); print(Fte[['prev1_d','next1_d','w10_n']].describe().loc[['mean','50%']])
