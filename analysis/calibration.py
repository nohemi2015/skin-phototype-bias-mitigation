import numpy as np, pandas as pd, re
from pathlib import Path
from sklearn.metrics import roc_auc_score
R=Path('/tmp/repo/results')
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def newton(X,y,offset=None,it=50):
    b=np.zeros(X.shape[1]); off=np.zeros(len(y)) if offset is None else offset
    for _ in range(it):
        eta=X@b+off; mu=1/(1+np.exp(-eta)); W=mu*(1-mu)+1e-12
        g=X.T@(y-mu); H=X.T@(X*W[:,None]); step=np.linalg.solve(H,g); b+=step
        if np.max(np.abs(step))<1e-9: break
    return b
def cal(y,p):
    lp=logit(p); n=len(y)
    slope=newton(np.c_[np.ones(n),lp],y)[1]
    citl=newton(np.ones((n,1)),y,offset=lp)[0]
    bins=np.clip((p*10).astype(int),0,9); ece=sum(abs(p[bins==k].mean()-y[bins==k].mean())*(bins==k).mean() for k in range(10) if (bins==k).any())
    return dict(brier=np.mean((p-y)**2),ece=ece,citl=citl,slope=slope,mean_p=p.mean(),prev=y.mean(),auc=roc_auc_score(y,p))
rows=[]
d7=pd.read_csv(R/'derm7pt'/'predicciones_derm7pt.csv')
for cfg in ['A','B','C40','D']:
  for arch in ['mobilenetv2','efficientnetb0']:
    for sfx,seed in [('',2026),('_s2027',2027),('_s2028',2028)]:
      tag=f'FINAL_{cfg}_{arch}{sfx}'
      c=pd.read_csv(R/'control_brownskin'/f'pred_{tag}_test_BS.csv'); rows.append(dict(cfg=cfg,arch=arch,seed=seed,cohort='Brown Skin (control)',**cal(c.y_true.values,c.y_score.values)))
      for n,lab in [('DDI','DDI'),('DDI2','DDI-2'),('Fitzpatrick','Fitzpatrick17k'),('PADUFES','PAD-UFES-20')]:
        d=pd.read_csv(R/'zeroshot'/tag/n/f'predicciones_zeroshot_{n}.csv'); rows.append(dict(cfg=cfg,arch=arch,seed=seed,cohort=lab,**cal(d.y_true.values,d.y_score.values)))
      for mod in ['dermoscopic','clinical']:
        d=d7[(d7.model==tag)&(d7.modality==mod)]
        assert len(d)==1003,(tag,mod,len(d))
        rows.append(dict(cfg=cfg,arch=arch,seed=seed,cohort=f'Derm7pt {mod}',**cal(d.y_true.values,d.y_score.values)))
r=pd.DataFrame(rows); r.to_csv('/tmp/new/calibration_per_model.csv',index=False)
m=r.groupby(['arch','cfg','cohort'],sort=False).agg(['mean','min','max']).drop(columns='seed')
m.to_csv('/tmp/new/calibration_seedmeans.csv')
pd.set_option('display.width',250)
print(r.groupby(['arch','cohort','cfg'],sort=False)[['brier','ece','citl','slope','mean_p','prev','auc']].mean().round(3).to_string())
