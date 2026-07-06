"""Spaceship Titanic V3 - DeepSeek-generated feature engineering."""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from datetime import datetime

train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/test.csv')

def engineer_all(df):
    df = df.copy()
    df['Cabin_deck'] = df['Cabin'].str.split('/').str[0].fillna('Unknown')
    df['Cabin_num'] = pd.to_numeric(df['Cabin'].str.split('/').str[1], errors='coerce').fillna(-1)
    df['Cabin_side'] = df['Cabin'].str.split('/').str[2].fillna('Unknown')
    df['Group'] = df['PassengerId'].str[:4]
    for c in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']: df[c] = df[c].fillna(0)
    df['TotalSpend'] = df['RoomService']+df['FoodCourt']+df['ShoppingMall']+df['Spa']+df['VRDeck']
    df['HasSpend'] = (df['TotalSpend']>0).astype(int)
    df['SpendPerService'] = df['TotalSpend']/(df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]>0).sum(axis=1).clip(1)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median() if df['Age'].dtype != object else 27)
    for c in ['VIP','CryoSleep']: df[c] = df[c].fillna(False)
    df['VIP_Age'] = df['VIP'].astype(int)*df['Age']
    df['CryoSleep_Spend'] = df['CryoSleep'].astype(int)*df['TotalSpend']
    for c in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
        df[f'{c}_ratio'] = df[c]/df['TotalSpend'].clip(1)
    df['GroupSize'] = df['Group'].map(df.groupby('Group').size())

    # DeepSeek V3: Group wealth features
    g = df.groupby('Group')
    df['group_total_spend'] = df['Group'].map(g['TotalSpend'].sum())
    df['group_avg_age'] = df['Group'].map(g['Age'].mean())
    child_count_map = g['Age'].apply(lambda x: (x < 12).sum()).to_dict()
    df['group_child_count'] = df['Group'].map(child_count_map)
    df['group_vip_ratio'] = df['Group'].map(g['VIP'].mean())
    df['group_has_cryo'] = df['Group'].map(g['CryoSleep'].max()).astype(int)

    # DeepSeek V3: Inconsistency flags
    df['cryo_but_spent'] = ((df['CryoSleep']==True) & (df['TotalSpend']>0)).astype(int)
    df['awake_zero_spend'] = ((df['CryoSleep']==False) & (df['TotalSpend']==0)).astype(int)
    df['spend_per_year'] = df['TotalSpend'] / (df['Age'] + 1)

    # DeepSeek V3: Cabin complexity
    deck_median = df.groupby('Cabin_deck')['Cabin_num'].transform('median')
    df['cabin_center_deviation'] = abs(df['Cabin_num'] - deck_median)
    df['side_dest'] = df['Cabin_side'].astype(str) + '_' + df['Destination'].astype(str)
    df['deck_vip_status'] = df['Cabin_deck'].astype(str) + '_VIP' + df['VIP'].astype(str)

    return df

train = engineer_all(train)
test = engineer_all(test)

target = 'Transported'
y = (train[target]==True).astype(int).values
cat_cols = ['HomePlanet','CryoSleep','Destination','VIP','Cabin_deck','Cabin_side','side_dest','deck_vip_status']
drop = [target,'PassengerId','Name','Cabin']

X = train.drop(columns=[c for c in drop if c in train.columns], errors='ignore')
Xt = test.drop(columns=[c for c in drop if c in test.columns], errors='ignore')

for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder(); le.fit(pd.concat([X[col].astype(str),Xt[col].astype(str)]))
        X[col] = le.transform(X[col].astype(str)); Xt[col] = le.transform(Xt[col].astype(str))

common = [c for c in X.columns if c in Xt.columns and X[c].dtype.name != 'category']
X = X[common].fillna(-1).astype(float).values; Xt = Xt[common].fillna(-1).astype(float).values
X = StandardScaler().fit_transform(X); Xt = StandardScaler().fit_transform(Xt)

print(f'V3 Features: {X.shape[1]} (V2 had 25)')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {}
configs = {
    'CB': CatBoostClassifier(iterations=800, learning_rate=0.02, depth=7, random_seed=42, verbose=False, thread_count=-1),
    'HGB': HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=None, random_state=42),
    'LGB': lgb.LGBMClassifier(n_estimators=800, learning_rate=0.02, num_leaves=95, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1),
}

for name, base_m in configs.items():
    oof = np.zeros(len(y)); tpred = np.zeros(len(Xt))
    for tr, val in skf.split(X, y):
        m = base_m.__class__(**base_m.get_params())
        m.fit(X[tr], y[tr])
        oof[val] = m.predict_proba(X[val])[:, 1]
        tpred += m.predict_proba(Xt)[:, 1] / 5
    acc = accuracy_score(y, (oof>0.5).astype(int))
    models[name] = {'oof': oof, 'test': tpred, 'acc': float(acc)}
    print(f'  {name}: OOF={acc:.6f}')

# Blend
best = 0; bw = None
for w0 in np.arange(0, 1.01, 0.05):
    for w1 in np.arange(0, 1.01-w0, 0.05):
        w2 = 1-w0-w1
        b = w0*models['CB']['oof'] + w1*models['HGB']['oof'] + w2*models['LGB']['oof']
        a = accuracy_score(y, (b>0.5).astype(int))
        if a > best: best = a; bw = (w0,w1,w2)

bt = bw[0]*models['CB']['test'] + bw[1]*models['HGB']['test'] + bw[2]*models['LGB']['test']
sub = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/sample_submission.csv')
sub['Transported'] = (bt > 0.5).astype(bool)

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = f'D:/桌面/codex/科研港科技/experiments/spaceship_titanic/v3_deepseek_{ts}'
os.makedirs(out, exist_ok=True)
sub.to_csv(f'{out}/submission.csv', index=False)

print(f'\nV3 BLEND: {best:.6f} (CB={bw[0]:.2f}, HGB={bw[1]:.2f}, LGB={bw[2]:.2f})')
print(f'vs V2 best 0.8163: {best-0.8163:+.6f}')
print(f'New features: group_total_spend, group_avg_age, group_child_count, group_vip_ratio, group_has_cryo, cryo_but_spent, awake_zero_spend, spend_per_year, cabin_center_deviation, side_dest, deck_vip_status')
print(f'Output: {out}')
