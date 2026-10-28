#!/usr/bin/env python3
"""COSTAR-CVD: a reproducible OFFLINE research simulation, not clinical software.

Run in Google Colab as a notebook cell, or `python COSTAR_CVD_colab.py`.
The fixed evaluation set is NEVER used to fit models or select thresholds/scenarios.
The file includes deliberately visible checks and does NOT force positive findings.
"""
from __future__ import annotations
import os, sys, json, hashlib, time, traceback, warnings, platform, zipfile
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, linprog
from scipy.special import expit, logit
from scipy.stats import beta
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, roc_curve, precision_recall_curve, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tqdm.auto import tqdm

SEED = 20261028
B_COLS = ['age','gender','chestpain','restingBP','serumcholestrol',
          'fastingbloodsugar','restingrelectro']
A_COLS = ['maxheartrate','exerciseangia','oldpeak','slope']
V_COL = 'noofmajorvessels'
COLS = ['patientid'] + B_COLS + A_COLS + [V_COL,'target']
CATEGORICAL = {'gender','chestpain','fastingbloodsugar','restingrelectro',
               'exerciseangia','slope','noofmajorvessels'}

@dataclass(frozen=True)
class Config:
    seed: int = SEED
    n_dev: int = 600
    n_opt: int = 200
    n_eval: int = 200
    n_outer: int = 5
    n_inner: int = 3
    n_meta: int = 5
    c_test: float = .055
    c_refer: float = .18
    c_miss: float = 1.0
    gamma: float = 1.25
    capacity_test: float = .40
    capacity_referral: float = .58
    maximum_joint_miss: float = .42
    group_penalty: float = 0.0
    alpha: float = .05
    bootstrap_replications: int = 1500
    n_grid_quantiles: int = 11
    smoke_test: bool = False


def nowstr(): return datetime.now(timezone.utc).isoformat()

def progress_msg(s): print(f'\n[{datetime.now().strftime("%H:%M:%S")}] {s}',flush=True)

class Progress:
    """A live progress bar; ETA is empirical and is not a runtime promise."""
    def __init__(self, total):
        self.p = tqdm(total=total, desc='COSTAR-CVD | overall', unit='steps',
                      bar_format='{desc}: {percentage:3.0f}%|{bar:20}| '
                                 '{n_fmt}/{total_fmt} [elapsed {elapsed}, ETA {remaining}]')
    def step(self, msg):
        self.p.update(1)
        self.p.set_postfix_str(msg[:55])
        print(f'Completed: {msg}',flush=True)
    def close(self): self.p.close()

def jsonwrite(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,default=str,allow_nan=False)

def export_table(df, output, name, caption, label=None, index=False):
    output=Path(output)
    (output/'tables').mkdir(parents=True,exist_ok=True)
    # NaNs are intentionally present for undefined quantities; no fabricated zero imputation.
    df.to_csv(output/'tables'/f'{name}.csv',index=index)
    try:
        tex=df.to_latex(index=index, na_rep='--',escape=True,
            caption=caption,label=label or f'tab:{name}', float_format=lambda x:f'{x:.4f}')
        (output/'tables'/f'{name}.tex').write_text(tex,encoding='utf-8')
    except Exception as e:
        print('LaTeX export warning:',name,e)

def visual_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
       'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
       'savefig.dpi':250,'figure.dpi':100,'pdf.fonttype':42,'axes.grid':True,
       'grid.alpha':.14,'axes.axisbelow':True,'legend.frameon':False})

def savefig(out,name,fig):
    folder=Path(out)/'figures';folder.mkdir(exist_ok=True,parents=True)
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(folder/f'{name}.png',bbox_inches='tight',dpi=270)
    plt.close(fig)

def prepare_data(path,out):
    df=pd.read_csv(path)
    if list(df.columns)!=COLS:
        raise ValueError(f'Unexpected dataset schema: {list(df.columns)}. Expected {COLS}')
    if len(df)!=1000: raise ValueError(f'Expected exactly 1000 rows, obtained {len(df)}')
    if df.patientid.isna().any() or df.patientid.duplicated().any():
        raise ValueError('Identifiers are not distinct/complete; reassess partition unit.')
    if df.target.isna().any() or not set(df.target.unique()) <= {0,1}:
        raise ValueError('Target must be coded 0/1 without missingness.')
    invalid = [c for c in B_COLS+A_COLS+[V_COL] if not pd.api.types.is_numeric_dtype(df[c])]
    if invalid: raise ValueError('Unexpected nonnumeric fields: '+str(invalid))
    df['serumcholestrol'] = df['serumcholestrol'].replace(0,np.nan)
    audit = {
        'n':int(len(df)),'n_columns_original':14,'positive':int(df.target.sum()),
        'negative':int(len(df)-df.target.sum()),'zero_cholesterol_recoded_as_missing':int(df.serumcholestrol.isna().sum()),
        'zero_cholesterol_positive_count':int(df.loc[df.serumcholestrol.isna(),'target'].sum()),
        'unique_patientid':int(df.patientid.nunique()),
        'target_by_slope':df.groupby('slope')['target'].agg(['count','sum']).reset_index().to_dict('records'),
        'dataset_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'source':'https://doi.org/10.17632/dzz48mvjht.1',
        'recode':'serumcholestrol==0 -> NaN, training-fold imputation + indicator',
        'interpretation':'retrospective status; no known clinical acquisition times; not causal'}
    jsonwrite(Path(out)/'data_audit.json',audit)
    audit_df=pd.DataFrame({'field':['N','positives','negatives','serumcholestrol zeros',
                                    'complete-row duplicates','unique patient IDs'],
         'value':[len(df),int(df.target.sum()),int((df.target==0).sum()),
             audit['zero_cholesterol_recoded_as_missing'],int(df.duplicated().sum()),df.patientid.nunique()]})
    export_table(audit_df,out,'data_audit','Dataset quality checks')
    return df,audit

def split_sets(df,cfg,out):
    # No stratification: independent binomial certification conditional on learning data.
    rng=np.random.default_rng(cfg.seed)
    inds=rng.permutation(len(df))
    ids={'development':inds[:cfg.n_dev],
         'optimization':inds[cfg.n_dev:cfg.n_dev+cfg.n_opt],
         'evaluation':inds[cfg.n_dev+cfg.n_opt:]}
    if len(ids['evaluation'])!=cfg.n_eval: raise AssertionError('Unexpected split size')
    for s,ii in ids.items():
        df.loc[ii,['patientid','target']].assign(row_index=ii).to_csv(Path(out)/f'partition_{s}.csv',index=False)
    rows=[{'partition':k,'n':len(v),'positive':int(df.iloc[v].target.sum()),
       'prevalence':float(df.iloc[v].target.mean())} for k,v in ids.items()]
    export_table(pd.DataFrame(rows),out,'partition_summary','Non-stratified fixed random partitions')
    return {k:df.iloc[v].reset_index(drop=True) for k,v in ids.items()}


def preprocessing(columns, use_indicator=True):
    cat=[x for x in columns if x in CATEGORICAL]
    num=[x for x in columns if x not in CATEGORICAL]
    # For sensitivity without indicator, remove it within the fold (never using held-out data).
    numpipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=use_indicator)),
                      ('scale',StandardScaler())])
    # Dense matrix supports all selected models, stable across sklearn versions.
    catpipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                      ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return ColumnTransformer([('num',numpipe,num),('cat',catpipe,cat)],
          remainder='drop',sparse_threshold=0.0)

def learner_specs(cols,seed,smoke=False,use_indicator=True):
    def pipe(model): return Pipeline([('features',preprocessing(cols,use_indicator)),('clf',model)])
    # Small, a-priori grids: nested resampling, rather than optimization on the holdout.
    rf_n=60 if smoke else 150
    iters=40 if smoke else 100
    specs={
      'logistic':(pipe(LogisticRegression(penalty='elasticnet',solver='saga',
             l1_ratio=.35,C=1.0,max_iter=2500,tol=1e-3,random_state=seed)),
             {'clf__C':[.1,1.0] if not smoke else [1.0],
              'clf__l1_ratio':[.15,.65] if not smoke else [.35]}),
      'forest':(pipe(RandomForestClassifier(n_estimators=rf_n,
            min_samples_leaf=3,max_features='sqrt',random_state=seed,n_jobs=1)),
            {'clf__min_samples_leaf':[2,7] if not smoke else [3],
             'clf__max_features':['sqrt',.8] if not smoke else ['sqrt']}),
      'boosting':(pipe(HistGradientBoostingClassifier(max_iter=iters,
             l2_regularization=1., learning_rate=.07,
             max_leaf_nodes=15,random_state=seed)),
             {'clf__max_leaf_nodes':[7,15] if not smoke else [7],
              'clf__l2_regularization':[.2,3.] if not smoke else [1.]}),
    }
    return specs

def fit_tuned(model,grid,X,y,seed,cfg):
    inner=StratifiedKFold(n_splits=cfg.n_inner,shuffle=True,random_state=seed)
    tune=GridSearchCV(model,grid,cv=inner,scoring='neg_brier_score',
         n_jobs=1,refit=True,error_score='raise')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=UserWarning)
        warnings.filterwarnings('ignore',category=FutureWarning)
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore',category=ConvergenceWarning)
        tune.fit(X,y)
    return tune.best_estimator_,tune.best_params_,float(-tune.best_score_)

def weight_fit(Q,y,lam):
    Q=np.asarray(Q,float);y=np.asarray(y,float);k=Q.shape[1]
    center=np.ones(k)/k
    def obj(w): return float(np.mean((Q@w-y)**2) + lam*np.sum((w-center)**2))
    def jac(w): return 2*Q.T@(Q@w-y)/len(y)+2*lam*(w-center)
    result=minimize(obj,center,jac=jac,method='SLSQP',
      bounds=[(0.,1.)]*k,constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1,
                         'jac':lambda w:np.ones_like(w)}],
      options={'ftol':1e-11,'maxiter':400})
    if not result.success: raise RuntimeError(f'Stack optimizer failed: {result.message}')
    w=np.clip(result.x,0,1); return w/w.sum()

def choose_lambda(Q,y,seed, cfg):
    vals=[0.,.0001,.001,.01,.05]
    meta=KFold(n_splits=cfg.n_meta,shuffle=True,random_state=seed)
    records=[]
    for val in vals:
        losses=[]
        for tr,va in meta.split(Q):
            w=weight_fit(Q[tr],y[tr],val)
            losses.append(brier_score_loss(y[va],Q[va]@w))
        records.append({'lambda':val,'brier_cv':np.mean(losses)})
    tab=pd.DataFrame(records).sort_values(['brier_cv','lambda'])
    return float(tab.iloc[0]['lambda']),pd.DataFrame(records)

def calibration_fit(raw,y):
    u=np.clip(np.asarray(raw,float),1e-6,1-1e-6)
    x=logit(u);y=np.asarray(y,int)
    def objective(z):
        zz=z[0]+z[1]*x
        # stable negative binomial log likelihood
        loss=float(np.mean(np.logaddexp(0,zz)-y*zz))
        p=expit(zz)
        return loss, np.array([np.mean(p-y),np.mean((p-y)*x)])
    res=minimize(lambda z:objective(z),[0.,1.],jac=True,bounds=[(None,None),(0.,30.)],
                 method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12})
    if not res.success: raise RuntimeError('Calibration failed: '+res.message)
    return np.asarray(res.x,float)

def apply_cal(raw,ab):
    return expit(ab[0]+ab[1]*logit(np.clip(np.asarray(raw,float),1e-6,1-1e-6)))

class TierModel:
    def __init__(self, name, cols, cfg, indicator=True):
        self.name=name;self.cols=cols;self.cfg=cfg;self.indicator=indicator
    def fit(self,dev,prog=None):
        X=dev[self.cols]; y=dev.target.to_numpy(int)
        Q=np.full((len(dev),3),np.nan)
        names=['logistic','forest','boosting']
        outer=KFold(n_splits=self.cfg.n_outer,shuffle=True,random_state=self.cfg.seed+111)
        self.fitlog=[]
        specs=learner_specs(self.cols,self.cfg.seed,self.cfg.smoke_test,self.indicator)
        for fold,(tr,va) in tqdm(enumerate(outer.split(X)),total=self.cfg.n_outer,
              desc=f'{self.name}: nested CV',leave=False):
            # Each outer validation fold has zero access to its own tuning or preprocessing.
            for k,name in enumerate(names):
                mdl,params,score=fit_tuned(*specs[name],X.iloc[tr],y[tr],
                                            self.cfg.seed+fold*17+k,self.cfg)
                Q[va,k]=mdl.predict_proba(X.iloc[va])[:,1]
                self.fitlog.append({'tier':self.name,'outer_fold':fold+1,'learner':name,
                                    'params':json.dumps(params),'inner_brier':score})
            if prog:prog.p.set_postfix_str(f'Fitting {self.name}: outer fold {fold+1}/{self.cfg.n_outer}')
        if not np.isfinite(Q).all():raise AssertionError('OOF predictions incomplete')
        self.lambda_,self.penalty_cv=choose_lambda(Q,y,self.cfg.seed+19,self.cfg)
        self.weights=weight_fit(Q,y,self.lambda_)
        self.cal=calibration_fit(Q@self.weights,y)
        self.oof=apply_cal(Q@self.weights,self.cal)
        self.models={}; self.tuning=[]
        for k,name in enumerate(names):
            mod,params,score=fit_tuned(*specs[name],X,y,self.cfg.seed+177+k,self.cfg)
            self.models[name]=mod
            self.tuning.append({'tier':self.name,'learner':name,'params':json.dumps(params),
                                'full_dev_inner_brier':score})
        self.models_order=names
        return self
    def predict_components(self,df):
        Q=np.column_stack([self.models[k].predict_proba(df[self.cols])[:,1]
                           for k in self.models_order]);
        return Q
    def predict(self,df):
        return apply_cal(self.predict_components(df)@self.weights,self.cal)


def fit_suite(dev,cfg,out,prog):
    modelB=TierModel('B',B_COLS,cfg).fit(dev,prog)
    prog.step('Cross-fitted, calibrated first-stage ensemble')
    modelS=TierModel('S',B_COLS+A_COLS,cfg).fit(dev,prog)
    prog.step('Cross-fitted, calibrated second-stage ensemble')
    fitlogs=pd.DataFrame(modelB.fitlog+modelS.fitlog)
    export_table(fitlogs,out,'outer_tuning','Nested cross-validation per outer fold')
    export_table(pd.DataFrame(modelB.tuning+modelS.tuning),out,'final_hyperparameters',
                 'Selected training-only final model hyperparameters')
    export_table(pd.concat([modelB.penalty_cv.assign(tier='B'),modelS.penalty_cv.assign(tier='S')]),
      out,'stack_penalty_cv','Cross-validated stack penalty selection')
    rows=[]
    for model in [modelB,modelS]:
        rows.append({'tier':model.name,'lambda':model.lambda_,
                     **{'weight_'+k:float(w) for k,w in zip(model.models_order,model.weights)},
                     'cal_intercept':float(model.cal[0]),'cal_slope':float(model.cal[1]),
                     'dev_oof_brier':brier_score_loss(dev.target,model.oof)})
    export_table(pd.DataFrame(rows),out,'ensemble_coefficients','Development-only ensemble parameters')
    return modelB,modelS


def support(z,gamma):
    """Exact sorting oracle (manuscript label eq:sortoracle), using the same weight set for each policy."""
    x=np.asarray(z,float).ravel();n=len(x)
    if gamma<1:raise ValueError('Gamma must be >= 1')
    if gamma==1: return float(x.mean())
    v=np.sort(x)[::-1]; s=n/(gamma+1); m=int(np.floor(s)); eta=s-m
    top=float(v[:m].sum())+(float(eta*v[m]) if m<n else 0.)
    return float(x.sum()/(gamma*n) + (gamma-1/gamma)*top/n)

def support_binary_counts(q,n,gamma):
    q=np.asarray(q,float)
    if gamma==1: return q/n
    return q/(gamma*n)+(gamma-1/gamma)*np.minimum(q,n/(gamma+1))/n

def validate_oracle(seed,out):
    rng=np.random.default_rng(seed); tests=[]
    for gam in [1.,1.25,1.5,2.]:
        for _ in range(7):
            z=rng.uniform(-1,3,size=19); n=len(z)
            res=linprog(-z,A_eq=np.ones((1,n)),b_eq=[1],
                bounds=[(1/(gam*n),gam/n)]*n,method='highs')
            if not res.success:raise AssertionError('LP oracle test failed')
            got=support(z,gam); brute=-res.fun
            tests.append({'Gamma':gam,'support_formula':got,'linear_program':brute,
                          'absolute_error':abs(got-brute)})
            if abs(got-brute)>2e-9:raise AssertionError('Sorting oracle does not agree with LP')
    export_table(pd.DataFrame(tests),out,'oracle_unit_tests','Exact robust oracle versus numerical LP')
    return max(x['absolute_error'] for x in tests)


def candidate_grid(pB,pS,cfg):
    q=np.linspace(.04,.96,cfg.n_grid_quantiles)
    B=np.unique(np.round(np.r_[0,np.quantile(pB,q),1],6))
    # Use separately the finite threshold values of the full-score population.
    S=np.unique(np.round(np.r_[0,np.quantile(pS,q),1],6))
    grid=np.array([(ell,h,t) for ell in B for h in B if ell<h for t in S],dtype=float)
    return grid

def policy_vectors(pB,pS,grid):
    L=grid[:,0,None];H=grid[:,1,None];D=grid[:,2,None]
    T=(pB[None,:]>L)&(pB[None,:]<H)
    R=(pB[None,:]>=H)|(T&(pS[None,:]>=D))
    assert T.shape==R.shape
    return T,R

def policy_single(pB,pS,theta):
    l,h,t=theta
    T=(pB>l)&(pB<h)
    R=(pB>=h)|(T&(pS>=t))
    return T.astype(int),R.astype(int)

def vector_support_loss(loss,gamma):
    if gamma==1:return loss.mean(axis=1)
    n=loss.shape[1];k=n/(gamma+1);m=int(np.floor(k));eta=k-m
    v=np.sort(loss,axis=1)[:,::-1]
    top=v[:,:m].sum(axis=1)
    if m<n:top=top+eta*v[:,m]
    return loss.sum(axis=1)/(gamma*n)+(gamma-1/gamma)*top/n

def all_policy_values(T,R,y,gender,cfg,gamma):
    n=R.shape[1]
    yt=y[None,:]
    M=(yt*(~R)).astype(float)
    LS=cfg.c_test*T+cfg.c_refer*R+cfg.c_miss*M
    rT=support_binary_counts(T.sum(axis=1),n,gamma)
    rR=support_binary_counts(R.sum(axis=1),n,gamma)
    rM=support_binary_counts(M.sum(axis=1),n,gamma)
    loss=vector_support_loss(LS,gamma)
    subgroup=[]
    for code in (0,1):
        mask=(gender==code)&(y==1)
        subgroup.append(np.where(mask.sum()>0,M[:,mask].mean(axis=1),np.nan))
    gap=np.abs(subgroup[0]-subgroup[1])
    penalty=np.where(np.isfinite(gap),gap,0.)
    objective=loss+cfg.group_penalty*penalty
    feasible=(rT<=cfg.capacity_test+1e-12)&(rR<=cfg.capacity_referral+1e-12)&(rM<=cfg.maximum_joint_miss+1e-12)
    return {'rT':rT,'rR':rR,'rM':rM,'loss':loss,'objective':objective,
            'fnr0':subgroup[0],'fnr1':subgroup[1],'gap':gap,'feasible':feasible,
            'T_nominal':T.mean(axis=1),'R_nominal':R.mean(axis=1),'M_nominal':M.mean(axis=1)}

def select_grid(grid,values):
    mask=values['feasible']
    if not np.any(mask):return None
    # lexicographic: objective, robust missed rate, test rate, ell, h, t
    ids=np.where(mask)[0]
    ix=np.lexsort((grid[ids,2],grid[ids,1],grid[ids,0],
         values['rT'][ids],values['rM'][ids],values['objective'][ids]))
    return int(ids[ix[0]])

def optimize_sequential(pB,pS,y,gender,cfg,gamma,grid=None):
    if grid is None:grid=candidate_grid(pB,pS,cfg)
    T,R=policy_vectors(pB,pS,grid)
    values=all_policy_values(T,R,y,gender,cfg,gamma)
    idx=select_grid(grid,values)
    return {'theta':None if idx is None else grid[idx].tolist(),
            'idx':idx,'grid':grid,'values':values,
            'n_candidates':len(grid),'n_feasible':int(np.sum(values['feasible']))}

def optimize_single_score(score,y,gender,cfg,gamma):
    grid=np.unique(np.round(np.r_[0,np.quantile(score,np.linspace(0,1,121)),1,1.000001],7))
    T=np.zeros((len(grid),len(y)),bool)
    R=score[None,:]>=grid[:,None]
    # Worst-case cost/miss and same resource constraints, no extra panel.
    vals=all_policy_values(T,R,y,gender,cfg,gamma)
    ids=np.where(vals['feasible'])[0]
    if not len(ids):
        # Explicitly flagged diagnostic comparator selected subject to capacity only.
        ids=np.where(vals['rR']<=cfg.capacity_referral+1e-12)[0]
        if not len(ids): raise AssertionError('Empty referral capacity feasible set')
        ok=False
    else:ok=True
    ix=ids[np.lexsort((grid[ids],vals['rM'][ids],vals['objective'][ids]))[0]]
    return {'threshold':float(grid[ix]),'feasible_all_constraints':bool(ok),
            'robust_miss_optimization':float(vals['rM'][ix]),
            'robust_referral_optimization':float(vals['rR'][ix]),
            'robust_loss_optimization':float(vals['loss'][ix])}

def consistent_uniform(patient_id,seed):
    # The pseudo-random test invitation depends on identifiers + seed, never outcomes.
    # Independent randomization is a *simulation*; identifiers are not predictors.
    return np.array([int.from_bytes(hashlib.blake2b(f'{seed}:{int(i)}'.encode(),
                    digest_size=8).digest(),'little')/(2**64) for i in patient_id])

def optimize_random_panel(opt_scores,y,gender,ids,q,cfg,gamma):
    B,S=opt_scores
    T=consistent_uniform(ids,cfg.seed+555)<q
    grid=np.unique(np.round(np.quantile(B,np.linspace(.0,1.,31)),6))
    tgrid=np.unique(np.round(np.quantile(S,np.linspace(.0,1.,31)),6))
    triples=np.array([(h,t) for h in np.r_[grid,1.] for t in tgrid])
    R=(B[None,:]>=triples[:,0,None]) | (T[None,:] & (S[None,:]>=triples[:,1,None]))
    Ts=np.broadcast_to(T[None,:],R.shape)
    vals=all_policy_values(Ts,R,y,gender,cfg,gamma)
    j=select_grid(np.column_stack([np.zeros(len(triples)),triples]),vals)
    if j is None:
        # Still report an explicitly nonfeasible comparison under referral capacity.
        ids0=np.where(vals['rR']<=cfg.capacity_referral+1e-12)[0]
        if len(ids0)==0: return {'theta':None,'feasible_all_constraints':False,'q':float(q)}
        j=ids0[np.argmin(vals['objective'][ids0])]
        feasible=False
    else: feasible=True
    return {'theta':triples[j].tolist(),'feasible_all_constraints':bool(feasible),
            'q':float(q),'robust_rT':float(vals['rT'][j]),
            'robust_rR':float(vals['rR'][j]),'robust_m':float(vals['rM'][j])}

def cp_upper(x,n,eta):
    if n==0 or x==n: return 1.0
    return float(beta.ppf(1-eta,x+1,n-x))

def cp_lower(x,n,eta):
    if n==0 or x==0:return 0.0
    return float(beta.ppf(eta,x,n-x+1))

def certificates(y,gender,T,R,cfg,out):
    """ONE frozen primary policy. Bonferroni J=5, one-sided exact CP bounds."""
    n=len(y); M=(y==1)&(R==0)
    rows=[]
    for nm,x,den,cap in [('test_use',int(T.sum()),n,cfg.capacity_test),
                          ('referral_use',int(R.sum()),n,cfg.capacity_referral),
                          ('joint_missed_case',int(M.sum()),n,cfg.maximum_joint_miss)]:
        upper=cp_upper(x,den,cfg.alpha/5)
        rows.append({'endpoint':nm,'numerator':x,'denominator':den,'rate':x/den,
            'one_sided_upper_99pct':upper,'tolerance':cap,
            'certified':bool(upper<=cap)})
    for g in (0,1):
        mask=(y==1)&(gender==g); den=int(mask.sum());x=int(M[mask].sum())
        upper=cp_upper(x,den,cfg.alpha/5)
        rows.append({'endpoint':f'FNR_gender_code_{g}','numerator':x,
            'denominator':den,'rate':x/den if den else np.nan,
            'one_sided_upper_99pct':upper,'tolerance':np.nan,'certified':np.nan})
    tab=pd.DataFrame(rows)
    export_table(tab,out,'primary_exact_certificates',
             'Primary frozen policy: simultaneous 95% familywise one-sided Clopper--Pearson bounds')
    return tab

def decision_metrics(y,T,R,cfg):
    y=np.asarray(y,int);T=np.asarray(T,int);R=np.asarray(R,int)
    TP=int(((y==1)&(R==1)).sum());FN=int(((y==1)&(R==0)).sum())
    TN=int(((y==0)&(R==0)).sum());FP=int(((y==0)&(R==1)).sum())
    n=len(y);ny=int(y.sum());nn=n-ny
    return {'tested':float(T.mean()),'referred':float(R.mean()),
        'joint_missed':float(FN/n),'FNR':FN/ny if ny else np.nan,
        'sensitivity':TP/ny if ny else np.nan,
        'specificity':TN/nn if nn else np.nan,
        'FPR':FP/nn if nn else np.nan,'TP':TP,'FP':FP,'TN':TN,'FN':FN,
        'mean_loss':float((cfg.c_test*T+cfg.c_refer*R+cfg.c_miss*y*(1-R)).mean()),
        'total_loss':float((cfg.c_test*T+cfg.c_refer*R+cfg.c_miss*y*(1-R)).sum())}

def score_metrics(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    out={'AUROC':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
         'AUPRC':average_precision_score(y,p) if (y==1).any() else np.nan,
         'Brier':brier_score_loss(y,p),'logloss':log_loss(y,p,labels=[0,1]),
         'predicted_positive_fraction':float(p.mean()),'observed_prevalence':float(np.mean(y))}
    # Descriptive calibration on EVALUATION only: not fed back to the risk model.
    # Logistic calibration slope estimated by maximum likelihood; no fitted score replacement.
    x=logit(p)
    def obj(z):
        zz=z[0]+z[1]*x
        f=float(np.mean(np.logaddexp(0,zz)-np.asarray(y)*zz))
        return f,np.array([np.mean(expit(zz)-y),np.mean((expit(zz)-y)*x)])
    try:
        fit=minimize(obj,[0.,1.],method='L-BFGS-B',jac=True,
                     bounds=[(-25,25),(-25,25)],options={'maxiter':300})
        out['calibration_intercept']=float(fit.x[0]) if fit.success else np.nan
        out['calibration_slope']=float(fit.x[1]) if fit.success else np.nan
    except Exception:
        out['calibration_intercept']=np.nan;out['calibration_slope']=np.nan
    return out

def bootstrapped_comparisons(y,reg,score,policies,cfg,out):
    """Exploratory paired bootstrap, fixed predictions and decisions; no re-fitting."""
    rng=np.random.default_rng(cfg.seed+700); n=len(y)
    pnames=['COSTAR_primary']+[x for x in policies if x!='COSTAR_primary']
    primary=policies['COSTAR_primary']
    L0=cfg.c_test*primary['T']+cfg.c_refer*primary['R']+cfg.c_miss*y*(1-primary['R'])
    rows=[]
    B=cfg.bootstrap_replications
    # One common resample per replicate across every comparator is a paired analysis.
    draws=rng.integers(0,n,size=(B,n))
    for name in pnames[1:]:
        test=policies[name]
        L1=cfg.c_test*test['T']+cfg.c_refer*test['R']+cfg.c_miss*y*(1-test['R'])
        di=L0-L1 # Negative means COSTAR lower loss
        diffs=di[draws].mean(axis=1)
        lo,hi=np.percentile(diffs,[2.5,97.5])
        rows.append({'comparator':name,'difference_Costar_minus_comparator':float(di.mean()),
                'bootstrap_low_95pct':float(lo),'bootstrap_high_95pct':float(hi),
                'replicates':B,'metric':'mean decision loss','direction':'negative=favors COSTAR',
                'primary_superior_at_95pct_exploratory':bool(hi<0)})
    if 'B_logistic' in score and 'B_stack' in score:
        # Also show probabilistic Brier comparison, recognizing different operating goals.
        a=(y-score['B_stack'])**2-(y-score['B_logistic'])**2
        low,high=np.percentile(a[draws].mean(axis=1),[2.5,97.5])
        rows.append({'comparator':'B_logistic [Brier risk score]',
          'difference_Costar_minus_comparator':float(a.mean()),'bootstrap_low_95pct':low,
          'bootstrap_high_95pct':high,'replicates':B,'metric':'Brier score difference',
          'direction':'negative=favors stack','primary_superior_at_95pct_exploratory':bool(high<0)})
    table=pd.DataFrame(rows)
    export_table(table,out,'paired_bootstrap_comparisons',
          'Exploratory paired bootstrap contrasts on fixed independent evaluation data')
    return table


def scenario_sweep(pB,pS,y,gender,cfg,out,progress,primary):
    grid=primary['grid'];T,R=policy_vectors(pB,pS,grid)
    cache={}
    gammas=[1.,1.25,1.5,2.]
    gammas=sorted(set(gammas+[cfg.gamma]))
    for gam in gammas:
        cache[gam]=all_policy_values(T,R,y,gender,cfg,gam)
    rows=[]; frontier=[]
    # Change constraints, NOT costs, without accessing the evaluation data.
    budgets=[(.25,.45,.28),(.35,.50,.35),(.40,.58,.42),(.50,.65,.50)]
    budgets=list(dict.fromkeys(budgets+[(cfg.capacity_test,cfg.capacity_referral,cfg.maximum_joint_miss)]))
    for gam in gammas:
        vals=cache[gam]
        for kt,kr,mu in budgets:
            feas=(vals['rT']<=kt+1e-12)&(vals['rR']<=kr+1e-12)&(vals['rM']<=mu+1e-12)
            feasids=np.where(feas)[0]
            if len(feasids):
                j=feasids[np.lexsort((grid[feasids,2],grid[feasids,1],grid[feasids,0],
                    vals['rT'][feasids],vals['rM'][feasids],vals['objective'][feasids]))[0]]
            else:j=None
            rows.append({'Gamma':gam,'budget_test':kt,'budget_referral':kr,
                 'joint_miss_tolerance':mu,'feasible':j is not None,
                 'n_feasible':int(len(feasids)),
                 'l':float(grid[j,0]) if j is not None else np.nan,
                 'h':float(grid[j,1]) if j is not None else np.nan,
                 't':float(grid[j,2]) if j is not None else np.nan,
                 'worst_case_loss':float(vals['loss'][j]) if j is not None else np.nan,
                 'worst_test':float(vals['rT'][j]) if j is not None else np.nan,
                 'worst_referral':float(vals['rR'][j]) if j is not None else np.nan,
                 'worst_joint_miss':float(vals['rM'][j]) if j is not None else np.nan})
    tab=pd.DataFrame(rows)
    export_table(tab,out,'robust_capacity_sensitivity',
          'Optimization-set robustness and resource scenarios; not evaluated for selection')
    # All available policies (may be dominated; output is a frontier dataset).
    for gam in gammas:
        vals=cache[gam]
        for j in range(len(grid)):
            frontier.append({'Gamma':gam,'l':grid[j,0],'h':grid[j,1],'t':grid[j,2],
                 'robust_test':vals['rT'][j],'robust_referral':vals['rR'][j],
                 'robust_joint_miss':vals['rM'][j],'robust_loss':vals['loss'][j],
                 'primary_constraints_feasible':bool(vals['feasible'][j])})
    pd.DataFrame(frontier).to_csv(Path(out)/'tables'/'complete_candidate_frontier.csv.gz',
                                 index=False,compression='gzip')
    progress.step('Robust sensitivity scenarios and complete policy frontier')
    return tab,cache


def plot_score_quality(y,probs,out):
    visual_style()
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for name in ['B_logistic','B_forest','B_boosting','B_stack','S_stack']:
        if name not in probs:continue
        p=probs[name]
        fpr,tpr,_=roc_curve(y,p)
        precision,recall,_=precision_recall_curve(y,p)
        ax[0].plot(fpr,tpr,label=f'{name}: {roc_auc_score(y,p):.3f}',lw=1.7)
        ax[1].plot(recall,precision,label=f'{name}: {average_precision_score(y,p):.3f}',lw=1.7)
    ax[0].plot([0,1],[0,1],'--',color='#666',lw=.9)
    ax[1].axhline(float(np.mean(y)),color='#666',ls='--',lw=.9)
    ax[0].set(xlabel='False-positive rate',ylabel='Sensitivity',title='Discrimination: ROC')
    ax[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall')
    for a in ax:a.set(xlim=(0,1),ylim=(0,1));a.legend(fontsize=8,loc='best')
    fig.tight_layout();savefig(out,'01_roc_precision_recall',fig)

    # Equal-frequency groups with variability bars, descriptive not a conformal guarantee.
    fig,ax=plt.subplots(figsize=(6.5,5))
    for name,col in [('B_logistic','#7794b6'),('B_stack','#21436a'),('S_stack','#bf653c')]:
        if name not in probs:continue
        p=probs[name];order=np.argsort(p)
        bins=np.array_split(order,10)
        xs=[];ys=[];errs=[]
        for b in bins:
            pp=float(p[b].mean()); n=len(b); obs=float(np.mean(y[b]));
            # Clopper-Pearson, two-sided at 95% for descriptive bin labels.
            k=int(y[b].sum());lo=cp_lower(k,n,.025);hi=cp_upper(k,n,.025)
            xs.append(pp);ys.append(obs);errs.append((obs-lo,hi-obs))
        ax.errorbar(xs,ys,yerr=np.asarray(errs).T,fmt='o-',label=name,
                     lw=1.35,ms=3,capsize=2,color=col,alpha=.9)
    ax.plot([0,1],[0,1],ls='--',c='k',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted risk (decile)',
           ylabel='Observed positive-label fraction',title='Evaluation-set calibration')
    ax.legend();fig.tight_layout();savefig(out,'02_calibration_deciles',fig)


def plot_policy_comparisons(metrics,out):
    visual_style()
    use=metrics[metrics['policy'].isin(['COSTAR_primary','Sequential_nominal',
                                      'B_logistic','B_forest','B_boosting','B_stack',
                                      'B_logistic_fixed_0p5','Random_panel'])].copy()
    if not len(use):return
    use=use.sort_values('policy')
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    x=np.arange(len(use));colors=['#195c80' if p=='COSTAR_primary' else '#9caab7' for p in use.policy]
    ax[0].bar(x-.23,use['tested'],width=.22,color='#4a8f90',label='Additional tests')
    ax[0].bar(x,use['referred'],width=.22,color='#4074a4',label='Referrals')
    ax[0].bar(x+.23,use['joint_missed'],width=.22,color='#bb7657',label='Joint missed cases')
    ax[0].axhline(.0,color='#aaa',lw=.8)
    ax[0].set_ylabel('Fraction of all records');ax[0].legend(fontsize=9)
    ax[1].bar(x,use['mean_loss'],color=colors)
    ax[1].set_ylabel('Mean scenario loss');ax[1].set_title('Fixed-loss decision comparison')
    for a in ax:
        a.set_xticks(x,use.policy,rotation=40,ha='right',fontsize=8)
    fig.suptitle('Independent evaluation: frozen operating policies')
    fig.tight_layout();savefig(out,'03_policy_comparison',fig)


def plot_certification(tab,cfg,out):
    p=tab.iloc[:3].copy()
    fig,ax=plt.subplots(figsize=(7.5,4.2))
    x=np.arange(3)
    ax.bar(x-.2,p['rate'],width=.32,color='#4c83a6',label='Observed rate')
    ax.bar(x+.15,p['one_sided_upper_99pct'],width=.32,color='#c68a63',
           label='Simultaneous one-sided upper bound')
    for j,cap in enumerate(p['tolerance']):
        ax.plot([j-.4,j+.4],[cap,cap],color='#a03131',lw=2)
        ax.text(j,cap+.013,f'limit {cap:.2f}',fontsize=8,color='#a03131',ha='center')
    ax.set_xticks(x,['Testing','Referral','Joint missed'])
    ax.set_ylabel('Probability / rate');ax.set_ylim(0,min(1,max(.65,max(p.one_sided_upper_99pct)+.12)))
    ax.legend(loc='upper left',fontsize=8)
    ax.set_title('Frozen primary policy: exact familywise risk bounds')
    fig.tight_layout();savefig(out,'04_exact_safety_certificates',fig)


def plot_frontier(sc,cache,grid,cfg,out):
    fig,ax=plt.subplots(1,2,figsize=(11,4.7))
    for gam in sorted(cache):
        val=cache[gam]
        valid=(val['rT']<=cfg.capacity_test)&(val['rR']<=cfg.capacity_referral)
        ax[0].scatter(val['rM'][valid],val['loss'][valid],s=6,alpha=.11,
                      label=f'$\\Gamma$={gam:g}')
    ax[0].axvline(cfg.maximum_joint_miss,color='#a03131',ls='--',lw=1.2)
    ax[0].set(xlabel='Robust joint missed-label probability',
          ylabel='Worst-case expected loss',title='Capacity-constrained attainable region')
    ax[0].legend(markerscale=2,fontsize=8)
    zz=sc[(sc.budget_test==cfg.capacity_test)&(sc.budget_referral==cfg.capacity_referral)&
          (sc.joint_miss_tolerance==cfg.maximum_joint_miss)]
    if len(zz):
        ax[1].plot(zz.Gamma,zz.worst_case_loss,'o-',color='#18576f',label='Robust loss')
        ax[1].set(xlabel=r'Robustness parameter $\Gamma$',
           ylabel='Optimized worst-case loss',title='Robustness sensitivity')
    fig.tight_layout();savefig(out,'05_robust_frontier',fig)


def plot_sequential_flow(y,T,R,pB,cfg,out,theta):
    fig,ax=plt.subplots(figsize=(9,4.7))
    n=len(y); labels=['Initial low risk\n(no panel, no referral)',
       'Direct referral\n(no panel)', 'Additional panel\nthen referral',
       'Additional panel\nthen no referral']
    groups=[(~T.astype(bool))&(~R.astype(bool)),
            (~T.astype(bool))&(R.astype(bool)),
            (T.astype(bool))&(R.astype(bool)),
            (T.astype(bool))&(~R.astype(bool))]
    counts=[int(x.sum()) for x in groups]
    positive=[int(y[x].sum()) for x in groups]
    neg=[a-b for a,b in zip(counts,positive)]
    loc=np.arange(4)
    ax.barh(loc,positive,color='#aa6b48',label='Positive labels')
    ax.barh(loc,neg,left=positive,color='#6999a5',label='Negative labels')
    ax.set_yticks(loc,labels);ax.invert_yaxis();ax.set_xlabel('Evaluation-set records')
    for i,v in enumerate(counts):ax.text(v+1,i,str(v),va='center',fontsize=9)
    ax.set_xlim(0,max(counts)*1.18+1)
    ax.legend();ax.set_title('Information-respecting paths and retrospective label counts')
    fig.tight_layout();savefig(out,'06_sequential_decision_paths',fig)
    return pd.DataFrame({'path':labels,'n':counts,'label_positive':positive,
                         'label_negative':neg})


def plot_group_rates(y,gender,R,out):
    fig,ax=plt.subplots(figsize=(6.8,4.4))
    codes=[0,1];rates=[];low=[];high=[];ns=[]
    for g in codes:
        mask=(y==1)&(gender==g);n=int(mask.sum());miss=int(((R==0)&mask).sum());
        p=miss/n if n else np.nan
        rates.append(p);low.append(cp_lower(miss,n,.025) if n else np.nan)
        high.append(cp_upper(miss,n,.025) if n else np.nan);ns.append(n)
    ax.bar(np.arange(2),rates,color=['#5f8ca4','#b68a6a'],width=.5)
    ax.errorbar(np.arange(2),rates,yerr=[np.asarray(rates)-np.asarray(low),
                       np.asarray(high)-np.asarray(rates)],fmt='none',capsize=5,color='black')
    ax.set_xticks([0,1],[f'Code {g}, positives n={n}' for g,n in zip(codes,ns)])
    ax.set(ylabel='Missed-positive fraction among label positives',
           title='Descriptive subgroup false-negative fractions (95% CP intervals)')
    ax.set_ylim(0,min(1,max([x for x in high if np.isfinite(x)]+[.1])+.13))
    fig.tight_layout();savefig(out,'07_group_false_negative',fig)

def evaluate_ablation(name,dev,opt,ev,modelB,modelS,cfg,out,columnsB,columnsS,
                      indicator=True,refitB=False):
    # All ablations are specified in advance and do not modify the primary models.
    if refitB:
        B=TierModel(f'{name}_B',columnsB,cfg,indicator).fit(dev)
    else:B=modelB
    S=TierModel(f'{name}_S',columnsS,cfg,indicator).fit(dev)
    bp_opt=B.predict(opt);sp_opt=S.predict(opt)
    bp_ev=B.predict(ev);sp_ev=S.predict(ev)
    retrospective_only=(name=='retrospective_majorvessels')
    pp=(None if retrospective_only else
        optimize_sequential(bp_opt,sp_opt,opt.target.to_numpy(int),
             opt.gender.to_numpy(int),cfg,cfg.gamma))
    row={'ablation':name,'uses_majorvessels':V_COL in columnsS,
          'B_features':','.join(columnsB),'S_features':','.join(columnsS),
          'n_feasible_policies_on_opt':np.nan if pp is None else pp['n_feasible'],
          'n_policy_candidates':np.nan if pp is None else pp['n_candidates'],
          'retro_only_no_policy':retrospective_only}
    yy=ev.target.to_numpy(int)
    row.update({'AUROC_B':score_metrics(yy,bp_ev)['AUROC'],
                'Brier_B':score_metrics(yy,bp_ev)['Brier'],
                'AUROC_S':score_metrics(yy,sp_ev)['AUROC'],
                'Brier_S':score_metrics(yy,sp_ev)['Brier']})
    if pp is not None and pp['theta'] is not None:
        T,R=policy_single(bp_ev,sp_ev,pp['theta'])
        mm=decision_metrics(yy,T,R,cfg)
        row.update({k:mm[k] for k in ['mean_loss','tested','referred','joint_missed','FNR']})
        row.update({'ell':pp['theta'][0],'h':pp['theta'][1],
                    't':pp['theta'][2], 'opt_feasible':True})
    else:
        row.update({'mean_loss':np.nan,'tested':np.nan,'referred':np.nan,
                 'joint_missed':np.nan,'FNR':np.nan,
                    'opt_feasible':np.nan if retrospective_only else False})
    return row


def worst_case_weights(z,gamma):
    """Attaining feasible weights for the manuscript bounded-density-ratio uncertainty set."""
    z=np.asarray(z,float);n=len(z)
    if gamma==1:return np.ones(n)/n
    w=np.ones(n)/(gamma*n)
    s=n/(gamma+1);m=int(np.floor(s));eta=s-m
    ranks=np.argsort(-z,kind='mergesort')
    extra=(gamma-1/gamma)/n
    w[ranks[:m]]+=extra
    if m<n:w[ranks[m]]+=eta*extra
    assert np.isclose(w.sum(),1,atol=1e-10)
    assert np.all(w>=1/(gamma*n)-1e-10) and np.all(w<=gamma/n+1e-10)
    return w


def case_mix_diagnostics(opt,pB,pS,theta,cfg,out):
    T,R=policy_single(pB,pS,theta)
    y=opt.target.to_numpy(int)
    cost=cfg.c_test*T+cfg.c_refer*R+cfg.c_miss*y*(1-R)
    weights=worst_case_weights(cost,cfg.gamma)
    assert abs(np.dot(cost,weights)-support(cost,cfg.gamma))<1e-9
    info=pd.DataFrame({'optimization_row':np.arange(len(opt)),'label':y,
             'additional_test':T,'referral':R,'individual_loss':cost,
             'uniform_weight':1/len(y),'adversarial_weight':weights})
    info.to_csv(Path(out)/'tables'/'adversarial_reweighting_primary.csv',index=False)
    ess=1/(weights**2).sum()
    audit=pd.DataFrame({'index':['Gamma','effective_sample_size','nominal_mean_loss',
                   'worst_case_mean_loss','nominal_missed_case_rate',
                   'adversarial_weighted_missed_case_rate'],
         'value':[cfg.gamma,ess,np.mean(cost),np.dot(cost,weights),
            np.mean(y*(1-R)),np.dot(y*(1-R),weights)]})
    export_table(audit,out,'case_mix_adversarial_diagnostics',
                 'Attaining reweighting for primary robust optimization loss')
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
    ax[0].scatter(cost,weights*len(cost),s=24,alpha=.7,color='#285b7e')
    ax[0].axhline(1,color='gray',ls='--',lw=1)
    ax[0].set(xlabel='Observed individual scenario loss',
              ylabel='Adversarial weight / uniform weight',
              title='Worst-case empirical reweighting')
    ax[1].bar(['Nominal','Worst-case'],[np.mean(cost),np.dot(cost,weights)],
              color=['#7994a7','#b57854'])
    ax[1].set(ylabel='Empirical mean loss',title='Robust optimization objective')
    fig.tight_layout();savefig(out,'09_case_mix_adversary',fig)
    return audit


def cost_and_fairness_sensitivity(pB,pS,y,gender,cfg,out):
    from dataclasses import replace
    grid=candidate_grid(pB,pS,cfg);T,R=policy_vectors(pB,pS,grid)
    rows=[]
    for ctest in [.025,.055,.10]:
        for crefer in [.10,.18,.30]:
            for penalty in [0.,.15]:
                for gamma in [1.,1.25,1.5,2.]:
                    scenario=replace(cfg,c_test=ctest,c_refer=crefer,
                                     group_penalty=penalty)
                    vals=all_policy_values(T,R,y,gender,scenario,gamma)
                    idx=select_grid(grid,vals)
                    rows.append({'Gamma':gamma,'c_test':ctest,'c_referral':crefer,
                        'c_miss':cfg.c_miss,'gender_disparity_penalty':penalty,
                        'feasible':idx is not None,
                        'best_objective':vals['objective'][idx] if idx is not None else np.nan,
                        'worst_loss':vals['loss'][idx] if idx is not None else np.nan,
                        'worst_test_rate':vals['rT'][idx] if idx is not None else np.nan,
                        'worst_referral_rate':vals['rR'][idx] if idx is not None else np.nan,
                        'worst_joint_miss':vals['rM'][idx] if idx is not None else np.nan,
                        'gender_fnr_gap':vals['gap'][idx] if idx is not None else np.nan,
                        'ell':grid[idx,0] if idx is not None else np.nan,
                        'h':grid[idx,1] if idx is not None else np.nan,
                        't':grid[idx,2] if idx is not None else np.nan})
    df=pd.DataFrame(rows)
    export_table(df,out,'loss_ratio_and_group_penalty_sensitivity',
          'Purely optimization-sample cost-ratio and gender-disparity penalty scenarios')
    z=df[(df.Gamma==cfg.gamma)&(df.gender_disparity_penalty==0)].copy()
    mat=z.pivot(index='c_test',columns='c_referral',values='best_objective')
    fig,ax=plt.subplots(figsize=(6.8,4.9))
    image=ax.imshow(mat.to_numpy(),cmap='Blues',aspect='auto')
    ax.set_xticks(np.arange(mat.shape[1]),[f'{c:.3f}' for c in mat.columns])
    ax.set_yticks(np.arange(mat.shape[0]),[f'{c:.3f}' for c in mat.index])
    ax.set(xlabel='Referral-loss ratio $c_R/c_M$',
       ylabel='Additional-test-loss ratio $c_T/c_M$',
       title=f'Optimization robust objective, $\\Gamma={cfg.gamma:g}$')
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v=mat.iloc[r,c]
            ax.text(c,r,'infeasible' if pd.isna(v) else f'{v:.3f}',
                    ha='center',va='center',fontsize=8,
                    color='white' if np.isfinite(v) and v>np.nanmedian(mat.to_numpy()) else 'black')
    fig.colorbar(image,ax=ax,label='Worst-case scenario loss')
    fig.tight_layout();savefig(out,'10_cost_ratio_sensitivity',fig)
    return df


def score_auc_bootstrap(y,scores,cfg,out):
    rng=np.random.default_rng(cfg.seed+991)
    n=len(y);B=cfg.bootstrap_replications
    rows=[]
    for name1,name2 in [('B_stack','B_logistic'),('S_stack','B_stack'),
                         ('S_stack','B_logistic')]:
        obs=roc_auc_score(y,scores[name1])-roc_auc_score(y,scores[name2])
        draws=[]
        for _ in range(B):
            ix=rng.integers(0,n,n)
            if len(np.unique(y[ix]))<2:continue
            draws.append(roc_auc_score(y[ix],scores[name1][ix])-
                        roc_auc_score(y[ix],scores[name2][ix]))
        ci=np.percentile(draws,[2.5,97.5]) if len(draws) else [np.nan,np.nan]
        rows.append({'score_1':name1,'score_2':name2,'AUROC_difference':obs,
              'bootstrap_low_95pct':ci[0],'bootstrap_high_95pct':ci[1],
              'replications':len(draws),'interpretation':'exploratory paired bootstrap'})
    export_table(pd.DataFrame(rows),out,'paired_auc_bootstrap',
                 'Exploratory paired AUROC comparisons on fixed predictions')
    return pd.DataFrame(rows)


def independent_robust_descriptive(y,policies,cfg,out):
    rows=[]
    for name,pol in policies.items():
        T,R=pol['T'],pol['R']
        M=y*(1-R)
        L=cfg.c_test*T+cfg.c_refer*R+cfg.c_miss*M
        for gam in [1.,1.25,1.5,2.]:
            rows.append({'policy':name,'Gamma':gam,
                         'empirically_reweighted_loss':support(L,gam),
                         'empirically_reweighted_test':support(T,gam),
                         'empirically_reweighted_referral':support(R,gam),
                         'empirically_reweighted_joint_miss':support(M,gam),
                         'status':'DESCRIPTIVE ONLY; not a population confidence bound'})
    tab=pd.DataFrame(rows)
    export_table(tab,out,'evaluation_reweighting_descriptive',
                'Exploratory bounded empirical reweighting on independent evaluation sample')
    fig,ax=plt.subplots(figsize=(7.5,4.7))
    for name in ['COSTAR_primary','Sequential_nominal','B_logistic','B_forest','B_stack']:
        z=tab[tab.policy==name]
        if len(z):ax.plot(z.Gamma,z.empirically_reweighted_loss,'o-',label=name,markersize=4)
    ax.set(xlabel=r'Support-reweighting sensitivity $\Gamma$',
           ylabel='Independent evaluation empirical worst-case loss',
           title='Exploratory case-mix sensitivity (not risk certification)')
    ax.legend(fontsize=8);fig.tight_layout();savefig(out,'11_heldout_reweighting_exploratory',fig)
    return tab


def create_report(out,config,audit,primary,models,metric_df,score_df,cert,boot,scenario,ablations,
                  oracle_error,elapsed):
    prim=metric_df.loc[metric_df.policy=='COSTAR_primary'].iloc[0]
    bas=metric_df.loc[metric_df.policy=='B_logistic'].iloc[0]
    lossdiff=float(prim.mean_loss-bas.mean_loss)
    msg='lower' if lossdiff<0 else 'higher'
    constraints = cert.iloc[:3]
    observed_all=bool(constraints.certified.all())
    lines=[
      '# COSTAR-CVD: reproducible statistical research output',
      '', f'Generated UTC: {nowstr()}',
      '', '**Interpretation:** OFFLINE, retrospective decision simulation; NOT clinical advice.',
      '', '## Frozen study design',
      '',f'- Random seed: {config.seed}; split: 600 development / 200 policy optimization / 200 evaluation, no stratification.',
      '- Stage B: age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar, restingrelectro.',
      '- Stage A: maxheartrate, exerciseangia, oldpeak, slope. noofmajorvessels excluded from all primary decisions.',
      '- Serum cholesterol zeros recoded as missing; imputation and indicator fitted in training folds only.',
      '- Cross-fitted stacking and sigmoid recalibration use only development labels; thresholds only optimization labels.',
      '- All statistics reported as independent evaluation once for a prespecified primary policy.',
      '', '## Primary decision setting', '',
      f'- Costs (test, referral, missed positive): {config.c_test}, {config.c_refer}, {config.c_miss}.',
      f'- Gamma = {config.gamma}; testing capacity = {config.capacity_test}; referral capacity = {config.capacity_referral}; joint miss limit = {config.maximum_joint_miss}.',
      f'- Candidate policies: {primary["n_candidates"]}; feasible on optimization set: {primary["n_feasible"]}.',
      f'- Selected thresholds (ell, h, t): {primary["theta"]}.',
      '', '## Independent evaluation', '',
      f'- COSTAR: mean loss {prim.mean_loss:.4f}, test use {prim.tested:.3f}, referral use {prim.referred:.3f}, joint missed fraction {prim.joint_missed:.3f}.',
      f'- B-only logistic: mean loss {bas.mean_loss:.4f}, test use {bas.tested:.3f}, referral use {bas.referred:.3f}, joint missed fraction {bas.joint_missed:.3f}.',
      f'- COSTAR loss is {abs(lossdiff):.4f} units {msg} than this logistic comparator; statistical uncertainty is in paired_bootstrap_comparisons.csv.',
      f'- All three simultaneous population-capacity safety bounds satisfy the selected tolerances: **{observed_all}**.',
      '- A point estimate below a tolerance is NOT a certificate; see one-sided upper confidence limits.',
      '', '## Robust optimization / validation notes', '',
      f'- Maximum numerical error in sorting-oracle versus LP unit tests: {oracle_error:.3g}.',
      '- Gamma reweights only observed optimization-set records and is NOT a confidence set for an unknown population.',
      '- LP equivalence is a computational identity; it does not establish clinical or statistical superiority.',
      '- The fixed holdout is not reused for choosing costs, Gamma, budget scenarios, feature access, or thresholds.',
      '- Policy-comparison bootstrap intervals are exploratory and do not replace the independent exact certificate.',
      '- Base-learner score metrics and policy metrics address different estimands.',
      '', '## Scope and limitations', '',
      '- The CSV provides recorded disease labels, not prospective cardiovascular outcomes or referral benefits.',
      '- The additional panel is retrospectively observed in the CSV, but its timing/availability is not established.',
      '- Strong slope outcome separation and outcome-linked cholesterol zero coding warrant caution.',
      '- Evaluation assumes IID draws from the same data-generating distribution; it does not test domain shift.',
      '- No actual measurement costs, clinician capacity, randomized testing, or treatment effects are observed.',
      '- Fairness comparisons use only recorded gender codes; underlying semantic validity is not known.',
      '', '## Output navigation', '',
      '- `tables/`: machine-readable CSV and manuscript-oriented TeX tables; includes optimization frontier.',
      '- `figures/`: 270-dpi PNG and vector PDF figures.',
      '- `predictions/`: held-out risk scores and decisions; no model tuning is performed on these labels.',
      '- `configuration.json`, `data_audit.json`, `README_RESULTS.md`, `package_versions.json`.',
      '', f'Elapsed computational time: {elapsed/60:.2f} minutes.',
    ]
    (Path(out)/'README_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def find_data(path_override=None):
    if path_override and Path(path_override).is_file():return str(Path(path_override))
    candidates=[Path('/content/Cardiovascular_Disease_Dataset.csv'),
                Path.cwd()/'Cardiovascular_Disease_Dataset.csv',
                Path('/mnt/data/Cardiovascular_Disease_Dataset.csv')]
    for f in candidates:
        if f.is_file():return str(f)
    # Colab interactive upload; only occurs if data is absent.
    try:
        from google.colab import files
        print('Please upload Cardiovascular_Disease_Dataset.csv from Mendeley.')
        uploaded=files.upload()
        for fname in uploaded:
            if Path(fname).name=='Cardiovascular_Disease_Dataset.csv':return str(Path(fname))
        raise FileNotFoundError('Upload must have the exact dataset filename.')
    except ImportError:
        raise FileNotFoundError('CSV not found. Place Cardiovascular_Disease_Dataset.csv '
                                'in the current directory or specify --file path.')


def package(out,zip_path):
    out=Path(out);zip_path=Path(zip_path)
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as zf:
        for f in sorted(out.rglob('*')):
            if f.is_file():zf.write(f,arcname=str(f.relative_to(out)))
    if not zip_path.exists() or zip_path.stat().st_size<1000:
        raise AssertionError('Results ZIP creation failed')
    # Ensure all entries are readable.
    with zipfile.ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Corrupt archive member: '+bad)
    return zip_path


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='COSTAR-CVD (retrospective research only)')
    parser.add_argument('--file',default=None,help='CSV path')
    parser.add_argument('--output',default=None,help='results folder')
    parser.add_argument('--smoke-test',action='store_true',help='FAST test; NOT manuscript grade')
    parser.add_argument('--skip-ablations',action='store_true',help='debugging only')
    parser.add_argument('--no-download',action='store_true',help='Disable Colab automatic ZIP download')
    args,_=parser.parse_known_args(argv)
    cfg=Config()
    if args.smoke_test:
        from dataclasses import replace
        cfg=replace(cfg,n_outer=3,n_inner=2,n_meta=3,
                    bootstrap_replications=90,n_grid_quantiles=6,smoke_test=True)
    dest=Path(args.output or ('/content/COSTAR_CVD_RESULTS' if Path('/content').exists()
                                  else str(Path.cwd()/'COSTAR_CVD_RESULTS')))
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'tables').mkdir(exist_ok=True)
    (dest/'figures').mkdir(exist_ok=True)
    (dest/'predictions').mkdir(exist_ok=True)
    jsonwrite(dest/'configuration.json',asdict(cfg))
    versions={'python':sys.version,'platform':platform.platform(),
      'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
      'sklearn':sklearn_version,'matplotlib':matplotlib.__version__,
      'tqdm':__import__('tqdm').__version__}
    jsonwrite(dest/'package_versions.json',versions)
    start=time.perf_counter();rng=np.random.default_rng(cfg.seed)
    ablation_names=[] if args.skip_ablations else ['minus_slope','minus_slope_oldpeak',
                                'omit_cholesterol','omit_cholesterol_indicator',
                                'retrospective_majorvessels']
    pbar=Progress(total=12+len(ablation_names))
    try:
        progress_msg('Reading and validating uploaded dataset')
        filepath=find_data(args.file)
        df,audit=prepare_data(filepath,dest)
        sets=split_sets(df,cfg,dest)
        dev,opt,ev=sets['development'],sets['optimization'],sets['evaluation']
        oracle_error=validate_oracle(cfg.seed,dest)
        pbar.step('Dataset split, leakage audit, numerical LP oracle validation')

        modelB,modelS=fit_suite(dev,cfg,dest,pbar) # 2 progress steps
        try:
            import joblib
            (dest/'models').mkdir(exist_ok=True)
            joblib.dump(modelB,dest/'models'/'B_risk_model.joblib',compress=3)
            joblib.dump(modelS,dest/'models'/'S_risk_model.joblib',compress=3)
        except Exception as err:
            print('Model serialization unavailable; fitted predictions remain saved:',err)
        yy=ev.target.to_numpy(int);yo=opt.target.to_numpy(int)
        go=opt.gender.to_numpy(int);ge=ev.gender.to_numpy(int)
        Bopt=modelB.predict(opt);Sopt=modelS.predict(opt)
        Bev=modelB.predict(ev);Sev=modelS.predict(ev)
        bo=modelB.predict_components(opt);be=modelB.predict_components(ev)
        scores_opt={'B_logistic':bo[:,0],'B_forest':bo[:,1],
                'B_boosting':bo[:,2],'B_stack':Bopt,'S_stack':Sopt}
        scores_ev={'B_logistic':be[:,0],'B_forest':be[:,1],
                'B_boosting':be[:,2],'B_stack':Bev,'S_stack':Sev}
        export_table(pd.DataFrame([{'score':k,**score_metrics(yy,p)} for k,p in scores_ev.items()]),
              dest,'independent_score_metrics','Evaluation-set probability-score assessment')
        pbar.step('All independent risk-score discrimination and calibration metrics')

        primary=optimize_sequential(Bopt,Sopt,yo,go,cfg,cfg.gamma)
        opt_info={'n_candidates':primary['n_candidates'],'n_feasible':primary['n_feasible'],
                   'gamma':cfg.gamma,'theta':primary['theta']}
        if primary['theta'] is None:
            # Do not pretend a capacity-violating policy is feasible.
            jsonwrite(dest/'infeasible_primary_optimization.json',opt_info)
            raise RuntimeError('No feasible primary threshold policy. Primary costs/capacities '
                  'were not secretly relaxed. Inspect candidate_frontier and scenario table '
                  'by adjusting prespecified research design before a new, fresh evaluation.')
        jsonwrite(dest/'selected_primary_policy.json',opt_info)
        pbar.step('Three-threshold constrained robust optimization on optimization-only data')

        scenario,cache=scenario_sweep(Bopt,Sopt,yo,go,cfg,dest,pbar,primary) # +1
        case_mix_diagnostics(opt,Bopt,Sopt,primary['theta'],cfg,dest)
        cost_and_fairness_sensitivity(Bopt,Sopt,yo,go,cfg,dest)
        pbar.step('Worst-case distributional weights, cost ratios and group penalties')
        # Frozen secondary policies. Model selection uses development/optimization only.
        policies={}
        T,R=policy_single(Bev,Sev,primary['theta'])
        policies['COSTAR_primary']={'T':T,'R':R,'opt_feasible':True}
        nominal=optimize_sequential(Bopt,Sopt,yo,go,cfg,1.)
        if nominal['theta'] is not None:
            t,r=policy_single(Bev,Sev,nominal['theta'])
            policies['Sequential_nominal']={'T':t,'R':r,'opt_feasible':True}
        baseline_opt={}
        for name in ['B_logistic','B_forest','B_boosting','B_stack']:
            result=optimize_single_score(scores_opt[name],yo,go,cfg,cfg.gamma)
            baseline_opt[name]=result
            policies[name]={'T':np.zeros(len(ev),dtype=int),
                             'R':(scores_ev[name]>=result['threshold']).astype(int),
                             'opt_feasible':result['feasible_all_constraints']}
        # Deliberately simple untuned-operating-point benchmark (NOT capacity matched).
        naiveRopt=(scores_opt['B_logistic']>=.5).astype(int)
        naiveMopt=yo*(1-naiveRopt)
        naive_ok=(support(naiveRopt,cfg.gamma)<=cfg.capacity_referral+1e-12
                  and support(naiveMopt,cfg.gamma)<=cfg.maximum_joint_miss+1e-12)
        policies['B_logistic_fixed_0p5']={
            'T':np.zeros(len(ev),dtype=int),
            'R':(scores_ev['B_logistic']>=.5).astype(int),
            'opt_feasible':bool(naive_ok)}
        # Random panel benchmark: independent pseudo-random invitation, frozen on opt.
        ptest=float(policy_single(Bopt,Sopt,primary['theta'])[0].mean())
        randomized=optimize_random_panel((Bopt,Sopt),yo,go,opt.patientid.to_numpy(),
                                          ptest,cfg,cfg.gamma)
        if randomized['theta'] is not None:
            h,t=randomized['theta']; rndT=(consistent_uniform(ev.patientid.to_numpy(),cfg.seed+555)<ptest)
            rndR=(Bev>=h)|(rndT&(Sev>=t))
            policies['Random_panel']={'T':rndT.astype(int),'R':rndR.astype(int),
                                     'opt_feasible':randomized['feasible_all_constraints']}
        pd.DataFrame([{'policy':k,**v} for k,v in baseline_opt.items()]).to_csv(
                         dest/'tables'/'baseline_optimization.csv',index=False)
        jsonwrite(dest/'secondary_policy_parameters.json',{'nominal_theta':nominal['theta'],
                     'random_panel':randomized,'baseline_thresholds':baseline_opt})

        # Evaluation of policies is pure application; no optimization or model re-fitting.
        metrics=[]
        for name,policy in policies.items():
            metrics.append({'policy':name,'optimization_feasible_all_constraints':policy['opt_feasible'],
                            **decision_metrics(yy,policy['T'],policy['R'],cfg)})
        tab=pd.DataFrame(metrics)
        export_table(tab,dest,'independent_policy_metrics','Pre-frozen evaluation of diagnostic decisions')
        cert=certificates(yy,ge,T,R,cfg,dest)
        subgroup=[]
        for name,pol in policies.items():
            for g in (0,1):
                positive=(ge==g)&(yy==1); n=int(positive.sum())
                missing=int((positive&(pol['R']==0)).sum())
                subgroup.append({'policy':name,'recorded_gender_code':g,
                    'positive_n':n,'miss_n':missing,'FNR':missing/n if n else np.nan})
        export_table(pd.DataFrame(subgroup),dest,'subgroup_FNR','Descriptive groupwise false-negative fractions')
        boot=bootstrapped_comparisons(yy,ge,scores_ev,policies,cfg,dest)
        score_auc_bootstrap(yy,scores_ev,cfg,dest)
        independent_robust_descriptive(yy,policies,cfg,dest)
        pbar.step('Frozen baselines, decision comparisons, paired bootstrap and exact certificates')

        pred=pd.DataFrame({'evaluation_row':np.arange(len(ev)),
            'patientid':ev.patientid.to_numpy(),
            'target':yy,'gender_code':ge,**scores_ev,
            'COSTAR_test':T,'COSTAR_referral':R})
        for name,pol in policies.items():
            if name!='COSTAR_primary':pred[name+'_test']=pol['T'];pred[name+'_referral']=pol['R']
        pred.to_csv(dest/'predictions'/'frozen_evaluation_scores_and_decisions.csv',index=False)
        optpred=pd.DataFrame({'optimization_row':np.arange(len(opt)),'patientid':opt.patientid,
                              'target':yo,'gender_code':go,**scores_opt})
        optpred.to_csv(dest/'predictions'/'optimization_scores.csv',index=False)

        plot_score_quality(yy,scores_ev,dest)
        plot_policy_comparisons(tab,dest)
        plot_certification(cert,cfg,dest)
        plot_frontier(scenario,cache,primary['grid'],cfg,dest)
        paths=plot_sequential_flow(yy,T,R,Bev,cfg,dest,primary['theta'])
        export_table(paths,dest,'evaluation_decision_paths','Primary sequential patient trajectories')
        plot_group_rates(yy,ge,R,dest)
        pbar.step('Professional vector figures and decision-flow tabulation')
        pbar.step('Descriptive adversarial and cost-scenario figures exported')

        # Prespecified information-timing and missing-data sensitivity experiments.
        spec=[('minus_slope',B_COLS,B_COLS+[x for x in A_COLS if x!='slope'],True,False),
          ('minus_slope_oldpeak',B_COLS,B_COLS+[x for x in A_COLS if x not in ('slope','oldpeak')],True,False),
          ('omit_cholesterol',[x for x in B_COLS if x!='serumcholestrol'],
             [x for x in B_COLS+A_COLS if x!='serumcholestrol'],True,True),
          ('omit_cholesterol_indicator',B_COLS,B_COLS+A_COLS,False,True),
          ('retrospective_majorvessels',B_COLS,B_COLS+A_COLS+[V_COL],True,False)]
        ablation_rows=[]
        for name,bcols,scols,ind,refitb in spec:
            if name not in ablation_names:continue
            progress_msg('Sensitivity refit: '+name)
            try:
                row=evaluate_ablation(name,dev,opt,ev,modelB,modelS,cfg,dest,
                          bcols,scols,indicator=ind,refitB=refitb)
                ablation_rows.append(row)
            except Exception as e:
                print('Ablation failed (recorded, not concealed):',name,str(e))
                ablation_rows.append({'ablation':name,'error':str(e)})
            export_table(pd.DataFrame(ablation_rows),dest,'feature_access_sensitivity',
                  'Prespecified sequential information-timing and missingness ablations')
            pbar.step('Prespecified sensitivity refit: '+name)
        # Deliberately flag downstream information use in ablation table.
        if ablation_rows:
            aft=pd.DataFrame(ablation_rows)
            if 'mean_loss' in aft:
                fig,ax=plt.subplots(figsize=(9,4.7))
                comp=[('primary',float(tab.loc[tab.policy=='COSTAR_primary','mean_loss'].iloc[0]))]
                comp +=[(r['ablation'],r.get('mean_loss',np.nan)) for r in ablation_rows]
                vals=[u[1] for u in comp]
                ax.bar(np.arange(len(comp)),vals,color=['#18576f']+['#9caab7']*(len(comp)-1))
                ax.set_xticks(np.arange(len(comp)),[v[0].replace('_','\n') for v in comp],fontsize=8)
                ax.set(ylabel='Mean scenario decision loss',
                    title='Prespecified ablations (separately fitted and optimized)')
                fig.tight_layout();savefig(dest,'08_feature_access_ablations',fig)

        elapsed=time.perf_counter()-start
        create_report(dest,cfg,audit,primary,(modelB,modelS),tab,None,
                      cert,boot,scenario,ablation_rows,oracle_error,elapsed)
        pbar.step('Reproducibility report and provenance metadata')
        # progress count accounts for final archive in an extra step.
        artifact=dest.parent/(dest.name+'.zip')
        z=package(dest,artifact)
        pbar.step('Single ZIP archive verified for CRC/readability')
        pbar.close()
        print('\nSUCCESS | Results:',z,'\nZIP size (MB):',round(z.stat().st_size/1e6,2),
              '\nElapsed minutes:',round((time.perf_counter()-start)/60,2),flush=True)
        if not args.no_download:
            try:
                from google.colab import files
                files.download(str(z))
            except ImportError:pass
        return str(z)
    except Exception as exc:
        pbar.close()
        error={'time':nowstr(),'type':type(exc).__name__,
               'message':str(exc),'traceback':traceback.format_exc()}
        jsonwrite(dest/'ERROR_DIAGNOSTICS.json',error)
        # Preserve partial outputs, including failed-constraint tables when available.
        try:
            emergency=dest.parent/(dest.name+'_PARTIAL_ON_ERROR.zip');package(dest,emergency)
            print('Partial diagnostic ZIP:',emergency)
        except Exception:pass
        raise

if __name__ == '__main__':
    main()
