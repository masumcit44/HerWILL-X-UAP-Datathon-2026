import numpy as np, pandas as pd, lightgbm as lgb, sys, json, time, warnings
warnings.filterwarnings('ignore')
exec(open('feats.py').read().split('# OOF features')[0])   # reuse loaders + nb_feats + text_meta
Ftr=pd.read_parquet('Ftr.parquet'); Fte0=pd.read_parquet('Fte.parquet')
nbcols=[c for c in Ftr.columns if c.startswith(('prev','next','w'))]
def mf1(yt,p):
    cm=np.bincount(yt*3+p,minlength=9).reshape(3,3); tp=np.diag(cm).astype(float)
    P=tp/np.maximum(cm.sum(0),1); R=tp/np.maximum(cm.sum(1),1); return float((2*P*R/np.maximum(P+R,1e-12)).mean())
def f1s(yt,p):
    cm=np.bincount(yt*3+p,minlength=9).reshape(3,3); tp=np.diag(cm).astype(float)
    P=tp/np.maximum(cm.sum(0),1); R=tp/np.maximum(cm.sum(1),1); return (2*P*R/np.maximum(P+R,1e-12))
cfg=sys.argv[1] if len(sys.argv)>1 else 'all'
drop={'all':[], 'noid_raw':['rid'], 'nonb':nbcols}[cfg]
cols=[c for c in Ftr.columns if c not in drop]
SEED=int(sys.argv[3]) if len(sys.argv)>3 else 42
cw=len(y)/(3*np.bincount(y)); wts=cw[y]
params=dict(objective='multiclass',num_class=3,learning_rate=0.05,num_leaves=31,min_data_in_leaf=40,feature_fraction=0.8,
            bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=2,seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)
ROUNDS=int(sys.argv[2]) if len(sys.argv)>2 else 400
oof=np.zeros((len(y),3)); tst=np.zeros((len(tids),3)); t0=time.time()
for k in range(5):
    trm=folds!=k; va=folds==k
    m=lgb.train(params,lgb.Dataset(Ftr.loc[trm,cols],y[trm],weight=wts[trm]),ROUNDS)
    oof[va]=m.predict(Ftr.loc[va,cols])
    pool=np.flatnonzero(folds!=k); Fte=Fte0.copy()
    Fte[nbcols]=nb_feats(tids,ids[pool],y[pool])[nbcols].values   # same neighbour density as training rows
    tst+=m.predict(Fte[cols])/5
    print(k,round(time.time()-t0),round(mf1(y[va],oof[va].argmax(1)),4),flush=True)
p=oof.argmax(1); fs=[mf1(y[folds==k],p[folds==k]) for k in range(5)]
print(cfg,'HONEST stacker',round(mf1(y,p),6),f1s(y,p).round(4),np.round(fs,4),'std',round(np.std(fs),4))
np.savez(f'stack_{cfg}_{ROUNDS}_s{SEED}.npz',oof=oof,test=tst,fold_ids=folds)
