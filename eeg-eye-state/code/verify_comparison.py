from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,f1_score
B=Path(__file__).resolve().parent;O=B/'comparison';R=B/'results'
p=json.loads((O/'protocol.json').read_text());m=json.loads((O/'metrics.json').read_text());a=json.loads((O/'analysis.json').read_text())
baseline=pd.read_csv(R/'predictions_oof.csv');all_errors=[]
for n,h in p['source_hashes'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
for name,v in m.items():
    d=pd.read_csv(O/(name+'_oof.csv'));assert len(d)==3681
    for col in ['subject','session','segment','start_row','end_row_exclusive','label','fold']:
        assert np.array_equal(d[col],baseline[col]),(name,col)
    assert d.groupby('subject').fold.nunique().max()==1
    assert np.isfinite(d.score_open).all()
    cm=pd.crosstab(d.label,d.prediction).reindex(index=[0,1],columns=[0,1],fill_value=0).to_numpy()
    assert cm.tolist()==v['cm'];assert abs(np.trace(cm)/cm.sum()-v['accuracy'])<1e-12
    assert abs(np.mean(np.diag(cm)/cm.sum(axis=1))-v['balanced_accuracy'])<1e-12
    assert abs(roc_auc_score(d.label,d.score_open)-v['auc'])<1e-12
    assert abs(f1_score(d.label,d.prediction,average='macro')-v['macro_f1'])<1e-12
    assert np.array_equal(d.correct,(d.label==d.prediction).astype(int))
    if name=='lr':assert np.array_equal(d.prediction,baseline.prediction)
    all_errors.append(d.label!=d.prediction)
    for fold in v['folds']:
        sub=d[d.fold==fold['fold']];assert abs(sub.correct.mean()-fold['accuracy'])<1e-12
for outer in m['stack']['nested_splits']:
    f=outer['outer'];tr=set(baseline[baseline.fold!=f].subject);te=set(baseline[baseline.fold==f].subject);validation=[]
    for inner in outer['inner']:
        it,iv=set(inner['train']),set(inner['validation'])
        assert it.isdisjoint(iv) and it|iv==tr and (it|iv).isdisjoint(te)
        validation.extend(iv)
    assert set(validation)==tr and len(validation)==len(tr)
assert int(np.logical_and.reduce(all_errors).sum())==a['all_wrong']
for pair in a['paired_vs_lr']:
    assert pair['fixed_baseline_errors']-pair['new_errors']==round((m[pair['method']]['accuracy']-m['lr']['accuracy'])*3681)
(O/'verification.json').write_text(json.dumps(dict(models_checked=5,windows_per_model=3681,source_hashes_match=True,
    outer_subject_overlap=0,stack_inner_subject_overlap=0,baseline_predictions_identical=True,
    metrics_recomputed=True,common_errors_reconciled=True),indent=2),encoding='utf-8')
print('Verified five model metrics, aligned windows, original hashes, and all inner/outer subject splits.')
