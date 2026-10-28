"""Resumen de MobileNetV2 con split por lesion y comparaciones pareadas (prueba t corregida
de Nadeau-Bengio para CV repetida, ratio n_val/n_train, Holm sobre las secundarias).
Uso: python comparaciones_pareadas.py <resultados_leakage_safe.csv> <carpeta_salida>"""
import pandas as pd, numpy as np
from scipy import stats
import sys
src, out = sys.argv[1], sys.argv[2]
d=pd.read_csv(src)
if 'arch' in d: d=d[d['arch'].fillna('mobilenetv2')=='mobilenetv2']
d=d[~d.model.astype(str).str.startswith('FINAL')]
d=d.drop_duplicates(['model','repeat','fold'],keep='last')
# coherencia
n=d.tn+d.fp+d.fn+d.tp
print('max |acc - (tp+tn)/n|', (d.accuracy-(d.tp+d.tn)/n).abs().max())
print('max |youden - (sens+spec-1)|',(d.youden-(d.sensitivity+d.specificity-1)).abs().max())
print('n_val==n', (d.n_val==n).all())
order=['A','B','C10','C20','C40','C60','C80','D','E']
m=d.groupby('model')[['auc','accuracy','sensitivity','specificity','precision','f1','youden']].mean().loc[order]
m['auc_sd']=d.groupby('model').auc.std().loc[order]; m['n']=d.groupby('model').size().loc[order]
print(m.round(3).to_string())
# mejor fold C40
c=d[d.model=='C40'].sort_values('auc',ascending=False).iloc[0]
print('mejor C40:',c.repeat,c.fold,round(c.auc,4),c.model_file)
# paired
w=d.pivot_table(index=['repeat','fold'],columns='model',values='auc')
ntr=d[d.model=='B'].n_train.mean(); nte=d[d.model=='B'].n_val.mean(); ratio=nte/ntr
print('ratio test/train',round(ratio,3))
comps=[('A','B'),('B','C10'),('B','C20'),('B','C40'),('B','C60'),('B','C80'),('C20','C40'),('C40','D'),('C40','E'),('B','D'),('B','E')]
rows=[]
for a,b in comps:
    diff=(w[b]-w[a]).values; k=len(diff)
    mu=diff.mean(); var=diff.var(ddof=1)
    se=np.sqrt((1/k+ratio)*var); t=mu/se; p=2*stats.t.sf(abs(t),k-1)
    q=stats.t.ppf(0.975,k-1)
    rep=(w[b]-w[a]).groupby(level=0).mean()
    rows.append(dict(A=a,B=b,dAUC=mu,lo=mu-q*se,hi=mu+q*se,p=p,wins=(diff>0).sum(),k=k,rep_min=rep.min(),rep_max=rep.max()))
r=pd.DataFrame(rows)
# Holm sobre todas excepto la primaria (B,C40)
sec=r[~((r.A=='B')&(r.B=='C40'))].sort_values('p')
mcount=len(sec); adj=[];run=0
for i,p in enumerate(sec.p): run=max(run,min(1,(mcount-i)*p)); adj.append(run)
sec=sec.assign(p_holm=adj)
r=r.merge(sec[['A','B','p_holm']],on=['A','B'],how='left')
pd.set_option('display.width',200)
print(r.round(4).to_string())
r.to_csv(f'{out}/comparaciones_pareadas.csv',index=False)
m.to_csv(f'{out}/resumen_sin_duplicados.csv')
print(d[d.model=='A'].n_train.describe())
