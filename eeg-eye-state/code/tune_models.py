"""Nested subject CV, broad search + local SVM refinement, train-only thresholds.
Run python eye_state_baseline/tune_models.py --workers 8
Every outer fold is frozen to the original baseline. No holdout labels enter search.
"""
from pathlib import Path
import argparse,json,time,hashlib,platform
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold,ParameterSampler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix
from scipy.stats import loguniform
from joblib import Parallel,delayed,parallel_config
from threadpoolctl import threadpool_limits
import joblib,sklearn
B=Path(__file__).resolve().parent;R=B/'results';O=B/'tuning';O.mkdir(exist_ok=True)
SEED=42;FAMILIES=['lr','svm','rf','hgb','svm_select']
BASE={
 'lr':dict(C=1.,weight_mode='balanced',features='all'),
 'svm':dict(C=1.,gamma='scale',weight_mode='balanced',features='all'),
 'rf':dict(max_depth=None,min_samples_leaf=5,max_features='sqrt',max_samples=1.,weight_mode='balanced',features='all'),
 'hgb':dict(max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,weight_mode='balanced',features='all')}

def weights(y,g):
    frame=pd.DataFrame({'y':y,'g':g});counts=frame.groupby(['g','y']).y.transform('count').to_numpy()
    return 1/counts/len(set(g))/2
def macroba(y,p,g):return float(np.dot(weights(y,g),(y==p).astype(float)))
def cols(names,mode):
    return [i for i,n in enumerate(names) if mode=='all' or (mode=='relative' and '_relative_' in n) or (mode=='logpower' and '_logpower_' in n)]
def create(family,param):
    q=param.copy();mode=q.pop('weight_mode');q.pop('features')
    cw='balanced' if mode=='balanced' else None
    if family in ['svm','svm_select']:clf=SVC(**q,kernel='rbf',probability=False,class_weight=cw,random_state=SEED,cache_size=200)
    elif family=='lr':clf=LogisticRegression(**q,class_weight=cw,max_iter=3000,random_state=SEED)
    elif family=='rf':clf=RandomForestClassifier(**q,n_estimators=300,class_weight=cw,n_jobs=1,random_state=SEED)
    elif family=='hgb':clf=HistGradientBoostingClassifier(**q,class_weight=cw,early_stopping=False,random_state=SEED)
    return Pipeline([('scaler',StandardScaler()),('classifier',clf)])
def fit(family,param,x,y,g):
    m=create(family,param)
    kw={'classifier__sample_weight':weights(y,g)*len(y)} if param['weight_mode']=='subject_balanced' else {}
    return m.fit(x,y,**kw)
def score(family,m,x):return m.decision_function(x) if family in ['svm','svm_select'] else m.predict_proba(x)[:,1]
def threshold_default(family):return 0. if family in ['svm','svm_select'] else .5
def thresholds(family):return np.linspace(-1,1,41) if family in ['svm','svm_select'] else np.linspace(.2,.8,31)
def unique(params):
    out=[];seen=set()
    for p in params:
        key=json.dumps(p,sort_keys=True)
        if key not in seen:out.append(p);seen.add(key)
    return out
def candidates(family):
    wm=['balanced','none','subject_balanced']
    if family=='lr':return [dict(C=float(c),weight_mode=w,features='all') for c in np.logspace(-4,3,15) for w in wm]
    if family in ['svm','svm_select']:
        params=[dict(C=c,gamma=g,weight_mode=w,features='all') for c,g,w in [(1.,'scale','balanced'),(.1,.01,'balanced'),(10.,.01,'balanced')]]
        random=list(ParameterSampler(dict(C=loguniform(1e-3,1e3),gamma=loguniform(1e-4,10),weight_mode=wm),n_iter=64,random_state=SEED))
        params+= [dict(p,features='all') for p in random]
        if family=='svm_select':return [dict(p,features=f) for f in ['relative','logpower'] for p in params]
        return unique(params)
    if family=='rf':space=dict(max_depth=[3,5,8,12,None],min_samples_leaf=[1,3,5,10,20,40],max_features=['sqrt',.5,1.],max_samples=[.6,.8,1.],weight_mode=wm)
    else:space=dict(max_iter=[100,200,400],learning_rate=[.025,.05,.1],max_leaf_nodes=[3,7,15,31],min_samples_leaf=[10,20,40,80],l2_regularization=[0.,1.,10.,100.],weight_mode=wm)
    return unique([BASE[family]]+[dict(p,features='all') for p in ParameterSampler(space,n_iter=48,random_state=SEED)])

def evaluate(family,param,x,y,g,names,splits):
    ix=cols(names,param['features']);x=x[:,ix];s=np.full(len(y),np.nan)
    with threadpool_limits(limits=1):
        for tr,va in splits:
            m=fit(family,param,x[tr],y[tr],g[tr]);s[va]=score(family,m,x[va])
    assert np.isfinite(s).all()
    pred=s>=threshold_default(family)
    return dict(param=param,inner_subject_ba=macroba(y,pred,g),inner_accuracy=float(accuracy_score(y,pred)),scores=s)

def metrics(y,p,s,g):
    cm=confusion_matrix(y,p,labels=[0,1])
    return dict(accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),
        subject_balanced_accuracy=macroba(y,p,g),macro_f1=float(f1_score(y,p,average='macro')),
        auc=float(roc_auc_score(y,s)),cm=cm.tolist(),recall_closed=float(cm[0,0]/cm[0].sum()),recall_open=float(cm[1,1]/cm[1].sum()))

def main(workers):
    df=pd.read_csv(R/'features.csv');old=pd.read_csv(R/'predictions_oof.csv');raw=json.loads((R/'results.json').read_text(encoding='utf-8'))
    names=raw['features'];x=df[names].to_numpy();y=df.label.to_numpy();g=df.subject.to_numpy();outer=old.fold.to_numpy()
    assert df[['subject','session','start_row']].equals(old[['subject','session','start_row']])
    protocol=dict(date='2026-11-17',outer_folds=5,inner_folds=4,seed=SEED,workers=workers,objective='Mean subject balanced accuracy, equal weight per subject and class',
        source_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['features.csv','predictions_oof.csv']},
        broad_candidates={f:candidates(f) for f in FAMILIES},thresholds={f:thresholds(f).tolist() for f in FAMILIES},
        svm_refinement='Best two broad trials: C and gamma multipliers [0.3,1,3], same weight/feature mode; scale gamma center 1/n_features',
        rules=['Preserve all original windows and outer subject splits','Select parameters on inner grouped OOF only',
        'Select threshold only after selecting parameters, using inner OOF, ties favor default','No outer scoring until all searches finish',
        'svm_select compares all-feature SVM trials with relative-only and logpower-only trials inside each outer training set',
        'Global search space is informed by past results; evaluation is exploratory, not an untouched external test'])
    protocol_path=O/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())['source_hashes']==protocol['source_hashes']
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for fold in range(1,6):
        tr=np.flatnonzero(outer!=fold);te=np.flatnonzero(outer==fold);xt,yt,gt=x[tr],y[tr],g[tr]
        splits=list(GroupKFold(n_splits=4).split(xt,yt,gt))
        audit=dict(outer=fold,train=sorted(set(gt)),test=sorted(set(g[te])),inner=[])
        assert set(gt).isdisjoint(g[te])
        for a,b in splits:
            assert set(gt[a]).isdisjoint(gt[b]);audit['inner'].append(dict(train=sorted(set(gt[a])),validation=sorted(set(gt[b]))))
        (O/f'fold{fold}_split.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        for family in FAMILIES:
            stem=f'{family}_fold{fold}'
            if (O/(stem+'.json')).exists() and (O/(stem+'.npz')).exists():continue
            start=time.perf_counter();params=candidates(family)
            print(f'FOLD {fold} {family}: searching {len(params)} candidates x 4 subject folds',flush=True)
            with parallel_config(backend='loky',inner_max_num_threads=1):
                trials=Parallel(n_jobs=workers,batch_size=1)(delayed(evaluate)(family,p,xt,yt,gt,names,splits) for p in params)
            if family in ['svm','svm_select']:
                top=sorted(trials,key=lambda t:-t['inner_subject_ba'])[:2];extra=[]
                for t in top:
                    p=t['param'];gamma=1/len(cols(names,p['features'])) if p['gamma']=='scale' else p['gamma']
                    for cm in [.3,1,3]:
                        for gm in [.3,1,3]:extra.append(dict(p,C=float(p['C']*cm),gamma=float(gamma*gm)))
                seen={json.dumps(p,sort_keys=True) for p in params};extra=[p for p in unique(extra) if json.dumps(p,sort_keys=True) not in seen]
                with parallel_config(backend='loky',inner_max_num_threads=1):trials+=Parallel(n_jobs=workers,batch_size=1)(delayed(evaluate)(family,p,xt,yt,gt,names,splits) for p in extra)
            if family=='svm_select':
                prev=json.loads((O/f'svm_fold{fold}.json').read_text());z=np.load(O/f'svm_fold{fold}.npz')
                trials.append(dict(param=prev['best_params'],inner_subject_ba=prev['best_inner_subject_ba'],inner_accuracy=prev['best_inner_accuracy'],scores=z['inner_scores']))
            best=sorted(trials,key=lambda t:-t['inner_subject_ba'])[0]
            default=threshold_default(family)
            ts=thresholds(family);t_scores=[macroba(yt,best['scores']>=t,gt) for t in ts]
            bi=max(range(len(ts)),key=lambda i:(t_scores[i],-abs(ts[i]-default)))
            chosen=float(ts[bi]);p=best['param'];ix=cols(names,p['features'])
            with threadpool_limits(limits=1):
                final=fit(family,p,xt[:,ix],yt,gt);test_scores=score(family,final,x[te][:,ix]);train_scores=score(family,final,xt[:,ix])
            record=dict(fold=fold,family=family,best_params=p,feature_count=len(ix),best_inner_subject_ba=best['inner_subject_ba'],
                best_inner_accuracy=best['inner_accuracy'],threshold=chosen,default_threshold=default,
                inner_subject_ba_after_threshold=t_scores[bi],threshold_curve=dict(thresholds=ts.tolist(),subject_ba=t_scores),
                train_accuracy=float(accuracy_score(yt,train_scores>=chosen)),seconds=time.perf_counter()-start,
                candidate_count=len(trials),search_trials=[{k:v for k,v in t.items() if k!='scores'} for t in trials])
            np.savez_compressed(O/(stem+'.npz'),inner_scores=best['scores'],inner_rows=tr,outer_rows=te,outer_scores=test_scores)
            joblib.dump(dict(pipeline=final,feature_indices=ix,feature_names=[names[i] for i in ix],threshold=chosen,params=p),O/(stem+'.joblib'))
            (O/(stem+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
            print(f'FOLD {fold} {family}: finished {len(trials)} candidates; inner subject-BA={best["inner_subject_ba"]:.4f}; elapsed={record["seconds"]:.1f}s',flush=True)
    summarize(df,old)

def summarize(df,old):
    y=df.label.to_numpy();g=df.subject.to_numpy();result={};baseline=pd.read_csv(B/'comparison/svm_oof.csv')
    for family in FAMILIES:
        z=np.zeros(len(y));pred=np.zeros(len(y),int);default_pred=pred.copy();logs=[]
        for fold in range(1,6):
            q=json.loads((O/f'{family}_fold{fold}.json').read_text());arr=np.load(O/f'{family}_fold{fold}.npz');i=arr['outer_rows'];z[i]=arr['outer_scores'];pred[i]=(z[i]>=q['threshold']).astype(int);default_pred[i]=(z[i]>=q['default_threshold']).astype(int)
            logs.append(dict(**{k:v for k,v in q.items() if k!='search_trials'},metrics=metrics(y[i],pred[i],z[i],g[i]),default_metrics=metrics(y[i],default_pred[i],z[i],g[i])))
        output=old[['subject','session','segment','start_row','end_row_exclusive','label','fold']].copy();output['prediction']=pred;output['score_open']=z;output['default_prediction']=default_pred;output['correct']=(y==pred).astype(int)
        output.to_csv(O/(family+'_oof.csv'),index=False,encoding='utf-8-sig')
        subjects=[dict(subject=s,n=int((g==s).sum()),**metrics(y[g==s],pred[g==s],z[g==s],g[g==s])) for s in sorted(set(g))]
        comparisons={}
        for reference in ['lr','svm',family if family!='svm_select' else 'svm']:
            ref=pd.read_csv(B/'comparison'/f'{reference}_oof.csv');stats=[]
            for s in sorted(set(g)):
                mask=g==s;stats.append([int((pred[mask]==y[mask]).sum())-int(ref.loc[mask,'correct'].sum()),int(mask.sum())])
            stats=np.array(stats);ix=np.random.default_rng(SEED).integers(0,len(stats),size=(3000,len(stats)));b=stats[ix].sum(axis=1)
            comparisons[reference]=dict(delta_accuracy=float((pred==y).mean()-ref.correct.mean()),ci95=np.quantile(b[:,0]/b[:,1],[.025,.975]).tolist(),
                fixed=int(((ref.prediction!=y)&(pred==y)).sum()),new_errors=int(((ref.prediction==y)&(pred!=y)).sum()))
        result[family]=dict(**metrics(y,pred,z,g),default_threshold_metrics=metrics(y,default_pred,z,g),folds=logs,subjects=subjects,
            comparisons=comparisons,train_accuracy_mean=float(np.mean([q['train_accuracy'] for q in logs])),total_seconds=sum(q['seconds'] for q in logs),
            inner_candidate_evaluations=sum(q['candidate_count'] for q in logs))
    (O/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FINAL',json.dumps({f:{k:result[f][k] for k in ['accuracy','balanced_accuracy','subject_balanced_accuracy','macro_f1','auc']} for f in FAMILIES}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();main(args.workers)
