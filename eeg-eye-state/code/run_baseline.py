"""Reproducible subject-independent baseline; never modifies source recordings.

Usage: python eye_state_baseline/run_baseline.py
Inputs are already filtered by the course provider. No repeat filter or ICA.
"""
from pathlib import Path
import json, hashlib, platform, sys
import numpy as np
import pandas as pd
import scipy
from scipy.signal import welch
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve)
import joblib

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)
FS, WINDOW, TRIM, SEED = 250, 500, 250, 42
BANDS = [('delta',1,4), ('theta',4,8), ('alpha',8,13), ('beta',13,30), ('gamma',30,40)]
EXCLUDED = {'S0004':'readme: poor data in both sessions',
            'S0038':'readme: poor data in both channels/sessions',
            'S0035':'readme: FP2 short circuit', 'S0050':'readme: FP2 short circuit',
            'S0055':'readme: FP2 short circuit'}
FEATURES = [f'{ch}_{kind}_{band}' for ch in ['FP1','FP2']
            for kind in ['logpower','relative'] for band,_,_ in BANDS]
CONFIG = dict(fs=FS,window_samples=WINDOW,window_seconds=2,overlap=0,
    trim_each_state_edge_seconds=1,quality_keep=[0,1],
    max_abs_uV=100,flat_std_uV=0.1,bands=BANDS,
    welch=dict(window='hann',nperseg=250,noverlap=125,nfft=250,detrend='constant',scaling='density'),
    classifier='StandardScaler + LogisticRegression(C=1, class_weight=balanced, max_iter=2000)',
    evaluation='GroupKFold(n_splits=5); subject as group; no tuning',
    label_mapping={'0':'closed','1':'open'}, positive_label=1,seed=SEED)

def metrics(y, pred, prob):
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,average='macro')),precision_open=float(precision_score(y,pred,zero_division=0)),
        recall_open=float(recall_score(y,pred,zero_division=0)),f1_open=float(f1_score(y,pred,zero_division=0)),
        roc_auc=float(roc_auc_score(y,prob)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED))

def main():
    records, inventory, segments, window_audit = [], [], [], []
    epochs, psds, examples = [], [], {}
    quality_counts = np.zeros((2,4), dtype=np.int64)
    counts = dict(total_samples=0,eye_samples=0,eligible_samples=0,trim_samples=0,tail_samples=0,
                  candidate_windows=0,quality_rejected=0,nonfinite_rejected=0,amplitude_rejected=0,flat_rejected=0)
    manifest=[]
    for file_index, path in enumerate(sorted((ROOT/'data_processed').glob('*/*/eeg_data_processed.csv'))):
        subject, session = path.parent.parent.name.replace('#',''), path.parent.name
        a = pd.read_csv(path,header=None,names=['FP1','FP2','label','quality']).to_numpy(dtype=float)
        assert a.shape[1] == 4
        assert np.isin(a[:,2],[0,1,3]).all(), f'Unexpected label: {path}'
        assert np.isin(a[:,3],[0,1,2,3]).all(), f'Unexpected quality: {path}'
        manifest.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        counts['total_samples'] += len(a)
        eye = np.isin(a[:,2],[0,1])
        counts['eye_samples'] += int(eye.sum())
        inventory.append(dict(subject=subject,folder=path.parent.parent.name,session=session,rows=len(a),
            seconds=len(a)/FS,closed_seconds=float((a[:,2]==0).sum()/FS),open_seconds=float((a[:,2]==1).sum()/FS),
            task_seconds=float((a[:,2]==3).sum()/FS),nonfinite_signal_values=int((~np.isfinite(a[:,:2])).sum()),
            exclusion=EXCLUDED.get(subject,'')))
        for label in [0,1]:
            for q in range(4):
                quality_counts[label,q] += int(((a[:,2]==label)&(a[:,3]==q)).sum())
        bounds=np.r_[0,np.flatnonzero(np.diff(a[:,2]))+1,len(a)]
        for segment_id,(start,end) in enumerate(zip(bounds[:-1],bounds[1:])):
            label=int(a[start,2])
            reason=EXCLUDED.get(subject,'')
            if subject=='S0021' and session=='session2' and label==0:
                reason='readme: severe eye movements in session2 closed-eye block'
            segments.append(dict(subject=subject,session=session,segment=segment_id,label=label,
                start_row=int(start),end_row_exclusive=int(end),seconds=float((end-start)/FS),exclusion=reason))
            if label not in [0,1] or reason:
                continue
            counts['eligible_samples'] += int(end-start)
            lo,hi=int(start+TRIM),int(end-TRIM)
            if hi<=lo:
                counts['trim_samples']+=int(end-start)
                continue
            counts['trim_samples']+=2*TRIM
            n=(hi-lo)//WINDOW
            counts['tail_samples']+=(hi-lo)%WINDOW
            for k in range(n):
                st=lo+k*WINDOW
                chunk=a[st:st+WINDOW]
                counts['candidate_windows']+=1
                reason='kept'
                if np.any(chunk[:,3]>1):
                    reason='quality'
                elif not np.isfinite(chunk[:,:2]).all():
                    reason='nonfinite'
                elif np.max(np.abs(chunk[:,:2]))>100:
                    reason='amplitude'
                elif np.any(np.std(chunk[:,:2],axis=0)<0.1):
                    reason='flat'
                meta=dict(subject=subject,session=session,segment=segment_id,start_row=st,
                          end_row_exclusive=st+WINDOW,label=label,quality_max=int(chunk[:,3].max()))
                window_audit.append(dict(**meta,status=reason))
                if reason!='kept':
                    counts[reason+'_rejected']+=1
                    continue
                # Each window stays within one event block. Do not concatenate over rejected data.
                x=chunk[:,:2].T.copy()
                x-=x.mean(axis=1,keepdims=True)
                f,p=welch(x,fs=FS,axis=-1,**CONFIG['welch'])
                powers=np.stack([np.trapezoid(p[:,(f>=low)&(f<=high)],f[(f>=low)&(f<=high)],axis=1)
                                 for _,low,high in BANDS],axis=1)
                total=np.trapezoid(p[:,(f>=1)&(f<=40)],f[(f>=1)&(f<=40)],axis=1)
                rel=powers/np.maximum(total[:,None],1e-12)
                assert np.allclose(rel.sum(axis=1),1), 'Band integration mismatch'
                features=np.concatenate([np.log10(np.maximum(powers,1e-12)),rel],axis=1).ravel()
                records.append({**meta,**dict(zip(FEATURES,features.tolist()))})
                epochs.append(x.astype('float32'))
                psds.append(p.astype('float32'))
                if subject=='S0001' and label not in examples:
                    examples[label]=dict(**meta,time=(np.arange(WINDOW)/FS).tolist(),signal=x[0].tolist())
        if (file_index+1)%20==0:
            print(f'Read {file_index+1} recordings; retained {len(records)} windows',flush=True)

    df=pd.DataFrame(records)
    X=df[FEATURES].to_numpy(); y=df.label.to_numpy(); groups=df.subject.to_numpy()
    assert np.isfinite(X).all()
    assert df.groupby('subject').label.nunique().min()==2, 'A subject lacks a class'
    assert counts['eligible_samples']==counts['trim_samples']+counts['tail_samples']+counts['candidate_windows']*WINDOW
    assert counts['candidate_windows']==len(df)+sum(counts[k] for k in ['quality_rejected','nonfinite_rejected','amplitude_rejected','flat_rejected'])
    probability=np.full(len(df),np.nan); prediction=np.full(len(df),-1); fold_id=np.zeros(len(df),dtype=int)
    dummy_pred=np.full(len(df),-1); dummy_prob=np.full(len(df),np.nan); folds=[]
    coefficients=[]
    for fold,(train,test) in enumerate(GroupKFold(n_splits=5).split(X,y,groups),1):
        assert set(groups[train]).isdisjoint(groups[test])
        model=make_model().fit(X[train],y[train])
        probability[test]=model.predict_proba(X[test])[:,1]
        prediction[test]=model.predict(X[test]); fold_id[test]=fold
        coefficients.append(model[-1].coef_[0])
        dummy=DummyClassifier(strategy='most_frequent').fit(X[train],y[train])
        dummy_pred[test]=dummy.predict(X[test]); dummy_prob[test]=dummy.predict_proba(X[test])[:,1]
        folds.append(dict(fold=fold,train_subjects=sorted(set(groups[train])),test_subjects=sorted(set(groups[test])),
            train_windows=len(train),test_windows=len(test),**metrics(y[test],prediction[test],probability[test])))
        print('Fold',fold,'accuracy',round(folds[-1]['accuracy'],4),flush=True)
    assert np.isfinite(probability).all() and np.all(fold_id>0)
    pred=df[['subject','session','segment','start_row','end_row_exclusive','label']].copy()
    pred['fold']=fold_id; pred['prediction']=prediction; pred['probability_open']=probability
    pred['correct']=(y==prediction).astype(int)
    subjects=[]
    for subject,g in pred.groupby('subject'):
        m=metrics(g.label,g.prediction,g.probability_open)
        subjects.append(dict(subject=subject,n=len(g),fold=int(g.fold.iloc[0]),closed=int((g.label==0).sum()),
                             open=int((g.label==1).sum()),**m))
    subject_df=pd.DataFrame(subjects)
    # Cluster bootstrap: people, not windows, are resampled. Fixed OOF predictions.
    per_subject=pred.groupby('subject').correct.agg(['sum','count']).to_numpy()
    rng=np.random.default_rng(SEED)
    selections=rng.integers(0,len(per_subject),size=(2000,len(per_subject)))
    boot=per_subject[selections].sum(axis=1)
    ci=np.quantile(boot[:,0]/boot[:,1],[0.025,0.975]).tolist()
    overall=metrics(y,prediction,probability)
    baseline=metrics(y,dummy_pred,dummy_prob)
    all_psd=np.stack(psds)
    spectra={}
    for label in [0,1]:
        # Average within person first, then across people so prolific subjects do not dominate plots.
        by_person=np.stack([all_psd[(groups==s)&(y==label)].mean(axis=0) for s in sorted(set(groups))])
        spectra[str(label)]={'mean':by_person.mean(axis=0).tolist(),
                            'median':np.median(by_person,axis=0).tolist()}
    alpha_subject=df.groupby(['subject','label'])[['FP1_relative_alpha','FP2_relative_alpha']].mean().mean(axis=1).unstack()
    inv=pd.DataFrame(inventory)
    summary=pd.read_csv(ROOT/'subjects_summary.csv')
    data_summary=dict(subjects=int(inv.subject.nunique()),sessions=len(inv),size_bytes=sum(m['bytes'] for m in manifest),
        included_subjects=int(df.subject.nunique()),included_sessions=int(df[['subject','session']].drop_duplicates().shape[0]),
        included_windows=len(df),class_counts={str(k):int(v) for k,v in df.label.value_counts().sort_index().items()},
        age_min=int(summary.age.min()),age_max=int(summary.age.max()),age_mean=float(summary.age.mean()),
        quality_counts_samples=quality_counts.tolist(),counts=counts,excluded_subjects=EXCLUDED)
    fpr,tpr,_=roc_curve(y,probability)
    report=dict(config=CONFIG,data=data_summary,metrics=overall,dummy=baseline,folds=folds,
        accuracy_cluster_bootstrap_ci95=ci,subject_accuracy_mean=float(subject_df.accuracy.mean()),
        subject_accuracy_median=float(subject_df.accuracy.median()),
        subject_accuracy_min=float(subject_df.accuracy.min()),subject_accuracy_max=float(subject_df.accuracy.max()),
        fold_accuracy_mean=float(np.mean([x['accuracy'] for x in folds])),
        fold_accuracy_sd=float(np.std([x['accuracy'] for x in folds],ddof=1)),
        subjects=subjects,features=FEATURES,coef_mean=np.mean(coefficients,axis=0).tolist(),
        frequency=f.tolist(),spectra=spectra,roc=dict(fpr=fpr.tolist(),tpr=tpr.tolist()),
        examples=examples,alpha_subjects=[dict(subject=s,closed=float(r[0]),open=float(r[1])) for s,r in alpha_subject.iterrows()],
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__))
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for name,frame in [('features',df),('predictions_oof',pred),('recording_inventory',inv),
        ('segments',pd.DataFrame(segments)),('window_audit',pd.DataFrame(window_audit)),('subject_metrics',subject_df)]:
        frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    np.savez_compressed(OUT/'clean_epochs.npz',X=np.stack(epochs),y=y,subjects=groups.astype(str),
        fs=FS,feature_names=np.array(FEATURES),row_order='same as features.csv')
    # Convenience deployment artifact. Its training accuracy is NOT the evaluation result.
    final_model=make_model().fit(X,y)
    joblib.dump(dict(model=final_model,features=FEATURES,config=CONFIG),OUT/'baseline_model.joblib')
    print(json.dumps(dict(data=data_summary,metrics=overall,dummy=baseline,ci95=ci),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
