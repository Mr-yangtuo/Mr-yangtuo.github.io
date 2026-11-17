"""Frozen outer subject folds; stage1 common models, stage2 exploratory ensembles.
Run: python eye_state_baseline/compare_models.py --stage 1 (then --stage 2)
No subject/window removal and no hyperparameter search after baseline.
"""
from pathlib import Path
import argparse, json, time, hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix,roc_curve
from threadpoolctl import threadpool_limits

B=Path(__file__).resolve().parent; R=B/'results'; O=B/'comparison'; O.mkdir(exist_ok=True)
SEED=42
PARAMS={
 'lr':dict(C=1,class_weight='balanced',max_iter=2000),
 'svm':dict(C=1,kernel='rbf',gamma='scale',class_weight='balanced',probability=False),
 'rf':dict(n_estimators=300,max_depth=None,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',n_jobs=4),
 'hgb':dict(max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1,class_weight='balanced',early_stopping=False),
 'stack':dict(bases=['lr','svm','rf'],inner_folds=4,meta='StandardScaler + LogisticRegression',meta_C=1,meta_class_weight='balanced')}

def model(name):
    if name=='lr': m=LogisticRegression(**PARAMS[name],random_state=SEED)
    elif name=='svm': m=SVC(**PARAMS[name],random_state=SEED)
    elif name=='rf': m=RandomForestClassifier(**PARAMS[name],random_state=SEED)
    elif name=='hgb': m=HistGradientBoostingClassifier(**PARAMS[name],random_state=SEED)
    return make_pipeline(StandardScaler(),m)

def score(m,x):
    return m.decision_function(x) if isinstance(m[-1],(SVC,LogisticRegression)) else m.predict_proba(x)[:,1]

def measure(y,p,s):
    cm=confusion_matrix(y,p,labels=[0,1])
    return dict(accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),
        macro_f1=float(f1_score(y,p,average='macro')),auc=float(roc_auc_score(y,s)),cm=cm.tolist(),
        recall_closed=float(cm[0,0]/cm[0].sum()),recall_open=float(cm[1,1]/cm[1].sum()))

def fit_stack(x,y,g,xt):
    z=np.full((len(y),3),np.nan); inner_records=[]
    for a,b in GroupKFold(n_splits=4).split(x,y,g):
        assert set(g[a]).isdisjoint(g[b])
        inner_records.append(dict(train=sorted(set(g[a])),validation=sorted(set(g[b]))))
        for j,name in enumerate(['lr','svm','rf']):
            fitted=model(name).fit(x[a],y[a]); z[b,j]=score(fitted,x[b])
    assert np.isfinite(z).all()
    meta=model('lr').fit(z,y)
    testz=[]; trainz=[]
    for name in ['lr','svm','rf']:
        fitted=model(name).fit(x,y)
        testz.append(score(fitted,xt));trainz.append(score(fitted,x))
    testz=np.column_stack(testz);trainz=np.column_stack(trainz)
    return meta.predict(testz),meta.predict_proba(testz)[:,1],float(accuracy_score(y,meta.predict(trainz))),inner_records,meta[-1].coef_[0].tolist()

def main(stage):
    df=pd.read_csv(R/'features.csv'); old=pd.read_csv(R/'predictions_oof.csv')
    baseline=json.loads((R/'results.json').read_text(encoding='utf-8'))
    names=baseline['features']; x=df[names].to_numpy();y=df.label.to_numpy();g=df.subject.to_numpy(); folds=old.fold.to_numpy()
    assert np.array_equal(df.label,old.label) and np.array_equal(df.start_row,old.start_row)
    assert df[['subject','session','start_row']].equals(old[['subject','session','start_row']])
    common=dict(n=len(y),subjects=len(set(g)),feature_names=names,parameters=PARAMS,
        source_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['features.csv','predictions_oof.csv','clean_epochs.npz']},
        outer_folds=[dict(fold=f,train_subjects=sorted(set(g[folds!=f])),test_subjects=sorted(set(g[folds==f]))) for f in range(1,6)])
    (O/'protocol.json').write_text(json.dumps(common,indent=2),encoding='utf-8')
    output=json.loads((O/'metrics.json').read_text()) if (O/'metrics.json').exists() else {}
    for name in (['lr','svm','rf'] if stage==1 else ['hgb','stack']):
        pred=np.zeros(len(y),int); scores=np.zeros(len(y)); rows=[]; nested=[]; weights=[]
        for f in range(1,6):
            tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
            assert set(g[tr]).isdisjoint(g[te])
            start=time.perf_counter()
            if name=='stack':
                pred[te],scores[te],train_acc,inner,w=fit_stack(x[tr],y[tr],g[tr],x[te]); nested.append(dict(outer=f,inner=inner));weights.append(w)
            else:
                fitted=model(name).fit(x[tr],y[tr]); pred[te]=fitted.predict(x[te]);scores[te]=score(fitted,x[te]);train_acc=float(accuracy_score(y[tr],fitted.predict(x[tr])))
            elapsed=time.perf_counter()-start
            row=dict(fold=f,train_accuracy=train_acc,fit_predict_seconds=elapsed,**measure(y[te],pred[te],scores[te]));rows.append(row)
            print(name,'fold',f,'accuracy',round(row['accuracy'],4),'seconds',round(elapsed,1),flush=True)
        if name=='lr': assert np.array_equal(pred,old.prediction), 'Baseline could not be reproduced'
        saved=old[['subject','session','segment','start_row','end_row_exclusive','label','fold']].copy()
        saved['prediction']=pred;saved['score_open']=scores;saved['correct']=(pred==y).astype(int)
        saved.to_csv(O/(name+'_oof.csv'),index=False,encoding='utf-8-sig')
        subjects=[]
        for s in sorted(set(g)):
            ix=(g==s); subjects.append(dict(subject=s,n=int(ix.sum()),**measure(y[ix],pred[ix],scores[ix])))
        stats=saved.groupby('subject').correct.agg(['sum','count']).to_numpy()
        ix=np.random.default_rng(SEED).integers(0,len(stats),size=(3000,len(stats)))
        bootstrap=stats[ix].sum(axis=1); ci=np.quantile(bootstrap[:,0]/bootstrap[:,1],[.025,.975]).tolist()
        fpr,tpr,_=roc_curve(y,scores)
        output[name]=dict(**measure(y,pred,scores),folds=rows,subjects=subjects,ci95=ci,
            subject_accuracy_mean=float(np.mean([s['accuracy'] for s in subjects])),
            subject_accuracy_min=float(min(s['accuracy'] for s in subjects)),
            training_seconds=float(sum(r['fit_predict_seconds'] for r in rows)),
            train_accuracy_mean=float(np.mean([r['train_accuracy'] for r in rows])),
            roc=dict(fpr=fpr.tolist(),tpr=tpr.tolist()),nested_splits=nested,meta_weights=weights)
        (O/'metrics.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    report_analysis(df,old,output)

def report_analysis(df,old,output):
    methods=list(output); y=df.label.to_numpy(); people=sorted(df.subject.unique()); predictions={}
    for m in methods: predictions[m]=pd.read_csv(O/(m+'_oof.csv'))
    bysubject=pd.DataFrame({m:predictions[m].groupby('subject').correct.mean() for m in methods})
    bysubject.to_csv(O/'subject_comparison.csv',encoding='utf-8-sig')
    failures={m:(predictions[m].prediction.to_numpy()!=y) for m in methods}
    pairs=[]
    for m in methods[1:]:
        # Paired cluster bootstrap preserves matching windows and person correlation.
        stat=np.array([[int(((~failures[m])&(df.subject==s)).sum())-int(((~failures['lr'])&(df.subject==s)).sum()),int((df.subject==s).sum())] for s in people])
        idx=np.random.default_rng(SEED).integers(0,len(stat),size=(3000,len(stat)));b=stat[idx].sum(axis=1)
        pairs.append(dict(method=m,delta_accuracy=output[m]['accuracy']-output['lr']['accuracy'],
            ci95=np.quantile(b[:,0]/b[:,1],[.025,.975]).tolist(),
            fixed_baseline_errors=int((failures['lr']&~failures[m]).sum()),new_errors=int((~failures['lr']&failures[m]).sum())))
    diagnostics=[]
    epoch=np.load(R/'clean_epochs.npz')['X']
    for s in people:
        ix=(df.subject==s).to_numpy(); a=df.loc[ix,['FP1_relative_alpha','FP2_relative_alpha']].mean(axis=1)
        labels=df.loc[ix,'label']; al0=float(a[labels==0].mean());al1=float(a[labels==1].mean())
        diagnostics.append(dict(subject=s,n=int(ix.sum()),alpha_closed=al0,alpha_open=al1,alpha_gap=al0-al1,
            quality1_fraction=float((df.loc[ix,'quality_max']==1).mean()),rms_mean=float(np.sqrt((epoch[ix]**2).mean(axis=-1)).mean()),
            all_models_wrong=int(np.logical_and.reduce([v[ix] for v in failures.values()]).sum()),
            accuracy={m:float(bysubject.loc[s,m]) for m in methods}))
    # Error breakdowns are descriptive, no tuning or test-time use of quality/subject identifiers.
    breakdown={}
    for m in methods:
        breakdown[m]={}
        for field in ['quality_max','session','label']:
            breakdown[m][field]=[]
            for val,ix in df.groupby(field).groups.items():
                ids=np.array(list(ix));breakdown[m][field].append(dict(value=str(val),n=len(ids),accuracy=float((~failures[m][ids]).mean())))
    details=dict(paired_vs_lr=pairs,diagnostics=diagnostics,breakdown=breakdown,
        all_wrong=int(np.logical_and.reduce(list(failures.values())).sum()),
        error_overlap={a:{b:int((failures[a]&failures[b]).sum()) for b in methods} for a in methods})
    (O/'analysis.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps({m:{k:output[m][k] for k in ['accuracy','balanced_accuracy','macro_f1','auc']} for m in methods}),flush=True)
    print('PAIRED',json.dumps(pairs),flush=True)
    print('WORST',bysubject.sort_values('lr').head(8).to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=[1,2],required=True)
    with threadpool_limits(limits=4): main(p.parse_args().stage)
