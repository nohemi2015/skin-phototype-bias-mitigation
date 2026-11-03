"""AUC de los modelos de validacion cruzada sobre las imagenes del hold-out de los modelos finales
(cada imagen evaluada por el modelo del fold en que fue validacion), comparado con los modelos finales.
Uso: python cv_en_holdout.py results/ split_ledger.csv salida.csv"""
import pandas as pd, numpy as np, os, sys
from sklearn.metrics import roc_auc_score
R=sys.argv[1]   # carpeta results/ del repositorio
L=pd.read_csv(sys.argv[2])   # split_ledger.csv (descomprimido de results/split_ledger.zip)
v=L[(L.source=='bs')&(L.partition=='val')].copy()
v['fn']=v.filepath.str.replace('\\','/').str.split('/').str[-1]
ctrl=pd.read_csv(f'{R}/control_brownskin/pred_FINAL_C40_efficientnetb0_test_BS.csv')
H=set(ctrl.filepath.str.replace('\\','/').str.split('/').str[-1])
print('holdout imgs',len(H))
rows=[]
for arch,tag in [('mobilenetv2',''),('efficientnetb0','_efficientnetb0')]:
    for cfg in ['A','B','C40','D']:
        allp=[]
        for r in (1,2,3):
            for f in range(1,6):
                p=pd.read_csv(f'{R}/predictions_cv/{arch}/pred_{cfg}{tag}_rep{r}_fold{f}.csv')
                s=v[(v.repeat==r)&(v.fold==f)]
                assert len(s)==len(p) and (s.label.values==p.y_true.values).all()
                d=p.assign(fn=s.fn.values,repeat=r,fold=f)
                allp.append(d)
        d=pd.concat(allp); dh=d[d.fn.isin(H)]; dn=d[~d.fn.isin(H)]
        rep=[roc_auc_score(x.y_true,x.y_score) for _,x in dh.groupby('repeat')]
        fold_h=[roc_auc_score(x.y_true,x.y_score) for _,x in dh.groupby(['repeat','fold'])]
        rep_n=[roc_auc_score(x.y_true,x.y_score) for _,x in dn.groupby('repeat')]
        fin=[]
        for sfx in ['','_s2027','_s2028']:
            c=pd.read_csv(f'{R}/control_brownskin/pred_FINAL_{cfg}_{arch}{sfx}_test_BS.csv'); fin.append(roc_auc_score(c.y_true,c.y_score))
        rows.append(dict(arch=arch,cfg=cfg,cv_on_holdout_imgs=np.mean(rep),cv_hold_rep_min=min(rep),cv_hold_rep_max=max(rep),
             cv_hold_fold_min=min(fold_h),cv_hold_fold_max=max(fold_h),cv_on_other_imgs=np.mean(rep_n),final_mean=np.mean(fin),final_min=min(fin),final_max=max(fin),n_hold=len(dh)//3))
out=pd.DataFrame(rows); pd.set_option('display.width',250); print(out.round(3).to_string(index=False))
out.to_csv(sys.argv[3],index=False)
