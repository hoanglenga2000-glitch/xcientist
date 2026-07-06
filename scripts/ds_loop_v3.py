"""DeepSeek Search Loop V3 - Improved prompt to avoid redundant features + GPU params."""
import subprocess, json, re, os, sys, warnings, hashlib, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
warnings.filterwarnings('ignore')

def call_ds(prompt):
    for attempt in range(3):
        try:
            r = subprocess.run(['curl','-s','-X','POST','http://127.0.0.1:8088/api/llm/deepseek/smoke',
                '-H','Content-Type: application/json','-d',json.dumps({'prompt':prompt})],
                capture_output=True, text=True, timeout=90)
            d = json.loads(r.stdout)
            if d.get('ok'): return d.get('content','')
        except: pass
        time.sleep(3)
    return 'ERR'

train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/test.csv')

def baseline(df):
    df = df.copy()
    df['Cabin_deck'] = df['Cabin'].str.split('/').str[0].fillna('Unknown')
    df['Cabin_num'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce').fillna(-1)
    df['Cabin_side'] = df['Cabin'].str.split('/').str[2].fillna('Unknown')
    df['Group'] = df['PassengerId'].str[:4]
    for c in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']: df[c]=df[c].fillna(0)
    df['TotalSpend'] = df['RoomService']+df['FoodCourt']+df['ShoppingMall']+df['Spa']+df['VRDeck']
    df['HasSpend'] = (df['TotalSpend']>0).astype(int)
    df['Age'] = pd.to_numeric(df['Age'],errors='coerce').fillna(27)
    for c in ['VIP','CryoSleep']: df[c]=df[c].fillna(False)
    df['VIP_Age'] = df['VIP'].astype(int)*df['Age']
    df['CryoSleep_Spend'] = df['CryoSleep'].astype(int)*df['TotalSpend']
    for c in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
        df[f'{c}_ratio'] = df[c]/df['TotalSpend'].clip(1)
    df['GroupSize'] = df['Group'].map(df.groupby('Group').size())
    df['MemberNum'] = df['PassengerId'].str.split('_').str[1].astype(int)
    df['IsFirstInGroup'] = (df['MemberNum'] == 1).astype(int)
    return df

def score(train_df, test_df):
    target = 'Transported'
    y = (train_df[target]==True).astype(int).values
    drop = [target,'PassengerId','Name','Cabin']
    cat_cols = ['HomePlanet','CryoSleep','Destination','VIP','Cabin_deck','Cabin_side']
    X = train_df.drop(columns=[c for c in drop if c in train_df.columns], errors='ignore')
    Xt = test_df.drop(columns=[c for c in drop if c in test_df.columns], errors='ignore')
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder(); le.fit(pd.concat([X[col].astype(str),Xt[col].astype(str)]))
            X[col] = le.transform(X[col].astype(str)); Xt[col] = le.transform(Xt[col].astype(str))
    cc = [c for c in X.columns if c in Xt.columns]
    X_arr = X[cc].fillna(-1).astype(float).values; Xt_arr = Xt[cc].fillna(-1).astype(float).values
    X_arr = StandardScaler().fit_transform(X_arr); Xt_arr = StandardScaler().fit_transform(Xt_arr)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oofs = {'CB': np.zeros(len(y)), 'HGB': np.zeros(len(y)), 'LGB': np.zeros(len(y))}
    tps = {'CB': np.zeros(len(Xt_arr)), 'HGB': np.zeros(len(Xt_arr)), 'LGB': np.zeros(len(Xt_arr))}
    for tr, val in skf.split(X_arr, y):
        cb = CatBoostClassifier(iterations=800,learning_rate=0.02,depth=7,random_seed=42,verbose=False,thread_count=-1)
        cb.fit(X_arr[tr],y[tr]); oofs['CB'][val]=cb.predict_proba(X_arr[val])[:,1]; tps['CB']+=cb.predict_proba(Xt_arr)[:,1]/5
        hgb = HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,max_depth=None,random_state=42)
        hgb.fit(X_arr[tr],y[tr]); oofs['HGB'][val]=hgb.predict_proba(X_arr[val])[:,1]; tps['HGB']+=hgb.predict_proba(Xt_arr)[:,1]/5
        lgb_m = lgb.LGBMClassifier(n_estimators=800,learning_rate=0.02,num_leaves=95,subsample=0.8,colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=-1)
        lgb_m.fit(X_arr[tr],y[tr]); oofs['LGB'][val]=lgb_m.predict_proba(X_arr[val])[:,1]; tps['LGB']+=lgb_m.predict_proba(Xt_arr)[:,1]/5
    scores = {m: accuracy_score(y,(o>0.5).astype(int)) for m,o in oofs.items()}
    best = 0; bw = None
    for w0 in np.arange(0,1.01,0.05):
        for w1 in np.arange(0,1.01-w0,0.05):
            w2 = 1-w0-w1
            b = w0*oofs['CB']+w1*oofs['HGB']+w2*oofs['LGB']
            a = accuracy_score(y,(b>0.5).astype(int))
            if a > best: best = a; bw = (w0,w1,w2)
    tp = bw[0]*tps['CB']+bw[1]*tps['HGB']+bw[2]*tps['LGB']
    return best, scores, bw, X_arr.shape[1], tp

b_train = baseline(train); b_test = baseline(test)
b_score, b_scores, b_bw, b_nf, b_tp = score(b_train, b_test)
print(f'BASELINE: Blend={b_score:.6f} CB={b_scores["CB"]:.6f} HGB={b_scores["HGB"]:.6f} LGB={b_scores["LGB"]:.6f}', flush=True)
best = b_score; best_code = None; best_tp = b_tp

used = ['Cabin_deck','Cabin_num','Cabin_side','GroupSize','TotalSpend','HasSpend','VIP_Age','CryoSleep_Spend','*_ratio','MemberNum','IsFirstInGroup','Group','HomePlanet','CryoSleep','Destination','VIP','Age']

for i in range(12):
    print(f'\n--- Iter {i+1}/12 (best={best:.6f}) ---', flush=True)
    prompt = f"""You are optimizing Spaceship Titanic Kaggle competition (accuracy metric, target=Transported).
Current 5-fold OOF: {best:.6f}. Target: 0.820.

ALREADY USED (SKIP): {', '.join(used)}

Generate def add_features(df) with 2-3 NOVEL ideas:
- Deck-to-Destination risk asymmetry
- Spending luxury vs basic imbalance ratios
- Age-CryoSleep-VIP triple interactions
- Cabin spatial proximity to ship center
- Group composition diversity (age spread, role mix)

Output ONLY Python code:"""

    resp = call_ds(prompt)
    resp = resp.replace('```python','').replace('```','')
    lines = resp.strip().split('\n')
    cl = []; started = False
    for line in lines:
        if line.strip().startswith('def '): started = True
        if started: cl.append(line)
    code = '\n'.join(cl) if cl else resp

    if 'def add_features' not in code:
        print(f'  SKIP', flush=True)
        continue

    idea = code[:140].replace('\n',' ')
    print(f'  {idea}', flush=True)

    try:
        exec(code, {'pd': pd, 'np': np})
        at = locals()['add_features'](b_train.copy())
        a_test = locals()['add_features'](b_test.copy())
        s, sc, bw, nf, tp = score(at, a_test)
        imp = s > best
        if imp: best = s; best_code = code; best_tp = tp
        mk = '>>> NEW BEST <<<' if imp else f'(-{best-s:.6f})'
        print(f'  Blend={s:.6f} CB={sc["CB"]:.6f} HGB={sc["HGB"]:.6f} LGB={sc["LGB"]:.6f} {mk}', flush=True)
        if imp:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out = f'D:/桌面/codex/科研港科技/experiments/spaceship_titanic/ds_{ts}'
            os.makedirs(out, exist_ok=True)
            sub = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/sample_submission.csv')
            sub['Transported'] = (best_tp > 0.5).astype(bool)
            sub.to_csv(f'{out}/submission.csv', index=False)
            with open(f'{out}/code.py','w') as f: f.write(code)
            with open(f'{out}/metrics.json','w') as f: json.dump({'blend':best,'scores':{m:float(v) for m,v in sc.items()},'weights':{f'w{i}':float(bw[i]) for i in range(len(bw))},'features':nf},f)
    except Exception as e:
        print(f'  ERR: {str(e)[:120]}', flush=True)

print(f'\nFINAL: {best:.6f} (start=0.8163, delta={best-0.8163:+.6f})', flush=True)
