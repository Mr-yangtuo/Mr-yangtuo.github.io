"""Validate selection, thresholds, folds and saved model predictions independently."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score
B=Path(__file__).resolve().parent;O=B/'tuning';R=B/'results'
df=pd.read_csv(R/'features.csv');old=pd.read_csv(R/'predictions_oof.csv');names=json.loads((R/'results.json').read_text(encoding='utf-8'))['features']
x=df[names].to_numpy();y=df.label.to_numpy();groups=df.subject.to_numpy();m=json.loads((O/'metrics.json').read_text());p=json.loads((O/'protocol.json').read_text())
for n,h in p['source_hashes'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
def mean_subject_ba(y,p,g):return np.mean([balanced_accuracy_score(y[g==s],p[g==s]) for s in np.unique(g)])
for k,v in m.items():
    pred=pd.read_csv(O/(k+'_oof.csv'))
    for col in ['subject','session','segment','start_row','end_row_exclusive','label','fold']:assert np.array_equal(pred[col],old[col])
    assert len(pred)==3681 and pred.groupby('subject').fold.nunique().max()==1
    assert abs((pred.label==pred.prediction).mean()-v['accuracy'])<1e-12
    assert abs(mean_subject_ba(y,pred.prediction.to_numpy(),groups)-v['subject_balanced_accuracy'])<1e-12
    assert abs(roc_auc_score(y,pred.score_open)-v['auc'])<1e-12
    assert abs(f1_score(y,pred.prediction,average='macro')-v['macro_f1'])<1e-12
    assert abs((pred.label==pred.default_prediction).mean()-v['default_threshold_metrics']['accuracy'])<1e-12
    for f in range(1,6):
        q=json.loads((O/f'{k}_fold{f}.json').read_text());split=json.loads((O/f'fold{f}_split.json').read_text());z=np.load(O/f'{k}_fold{f}.npz');tr=z['inner_rows'];te=z['outer_rows']
        assert np.all(old.fold.iloc[tr]!=f) and np.all(old.fold.iloc[te]==f)
        assert set(groups[tr])==set(split['train']) and set(groups[te])==set(split['test']) and set(split['train']).isdisjoint(split['test'])
        val_people=[]
        for ins in split['inner']:
            a=set(ins['train']);b=set(ins['validation']);assert a.isdisjoint(b) and a|b==set(split['train']);val_people+=ins['validation']
        assert len(val_people)==len(set(val_people))==len(split['train'])
        assert q['best_inner_subject_ba']==max(t['inner_subject_ba'] for t in q['search_trials'])
        assert abs(mean_subject_ba(y[tr],z['inner_scores']>=q['default_threshold'],groups[tr])-q['best_inner_subject_ba'])<1e-12
        assert abs(mean_subject_ba(y[tr],z['inner_scores']>=q['threshold'],groups[tr])-max(q['threshold_curve']['subject_ba']))<1e-12
        saved=joblib.load(O/f'{k}_fold{f}.joblib');model=saved['pipeline'];inputs=x[te][:,saved['feature_indices']]
        s=model.decision_function(inputs) if k.startswith('svm') else model.predict_proba(inputs)[:,1]
        assert np.allclose(s,z['outer_scores']) and np.array_equal(s>=q['threshold'],pred.prediction.iloc[te])
        assert np.array_equal(s>=q['default_threshold'],pred.default_prediction.iloc[te])
result=dict(families=5,outer_fits=25,rows_per_model=3681,source_hashes_match=True,subject_overlap=0,
    search_choice_verified=True,inner_thresholds_verified=True,saved_models_predictions_verified=True,metrics_recomputed=True)
(O/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
