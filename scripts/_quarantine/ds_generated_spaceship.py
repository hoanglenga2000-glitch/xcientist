```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/train.csv')
test = pd.read_csv('D:/桌面/codex/科研港科技/tasks/spaceship_titanic/data/test.csv')

# Preprocessing
def preprocess(df):
    df = df.copy()
    
    # Cabin features
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['Side'] = df['Side'].fillna('Unknown')
    
    # Group aggregation features
    group_cols = ['HomePlanet', 'Deck', 'Side']
    for col in group_cols:
        if col in df.columns:
            df[f'{col}_count'] = df.groupby(col)['PassengerId'].transform('count')
            df[f'{col}_mean_Spending'] = df.groupby(col)['TotalSpending'].transform('mean')
    
    # Spending features
    df['TotalSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    df['SpendingRatio'] = df['TotalSpending'] / (df['TotalSpending'].mean() + 1e-8)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    
    # Fill missing values
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum', 'TotalSpending']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Encode categoricals
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    
    return df

# Prepare data
train_processed = preprocess(train)
test_processed = preprocess(test)

feature_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 
                'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNum', 'Side',
                'Deck_count', 'Deck_mean_Spending', 'Side_count', 'Side_mean_Spending',
                'TotalSpending', 'SpendingRatio', 'HasSpending']

X = train_processed[feature_cols]
y = train_processed['Transported'].astype(int)
X_test = test_processed[feature_cols]

# 5-fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
models = {
    'catboost': CatBoostClassifier(verbose=0, random_state=42, n_estimators=500, learning_rate=0.05),
    'lightgbm': LGBMClassifier(verbose=-1, random_state=42, n_estimators=500, learning_rate=0.05),
    'xgboost': XGBClassifier(verbose=0, random_state=42, n_estimators=500, learning_rate=0.05),
    'hgb': HistGradientBoostingClassifier(random_state=42, max_iter=500, learning_rate=0.05)
}

# Store OOF predictions and test predictions
oof_preds = {name: np.zeros(len(X)) for name in models.keys()}
test_preds = {name: np.zeros(len(X_test)) for name in models.keys()}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f'Fold {fold+1}')
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        oof_preds[name][val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds[name] += model.predict_proba(X_test)[:, 1] / 5

# Optimize blend weights
def objective(weights):
    weights = np.array(weights)
    blended = np.zeros(len(X))
    for i, name in enumerate(models.keys()):
        blended += weights[i] * oof_preds[name]
    return -accuracy_score(y, (blended > 0.5).astype(int))

# Initial weights
initial_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1) for _ in range(4)]
constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

result = minimize(objective, initial_weights, bounds=bounds, constraints=constraint, method='SLSQP')
optimal_weights = result.x

print(f'Optimal weights: {dict(zip(models.keys(), optimal_weights))}')

# Final OOF prediction
final_oof = np.zeros(len(X))
for i, name in enumerate(models.keys()):
    final_oof += optimal_weights[i] * oof_preds[name]
final_oof_acc = accuracy_score(y, (final_oof > 0.5).astype(int))
print(f'Final OOF Accuracy: {final_oof_acc:.4f}')

# Test prediction
final_test = np.zeros(len(X_test))
for i, name in enumerate(models.keys()):
    final_test += optimal_weights[i] * test_preds[name]
test_preds_binary = (final_test > 0.5).astype(bool)

# Save submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_preds_binary
})
submission.to_csv('D:/桌面/codex/科研港科技/experiments/spaceship_titanic/ds_gen/submission.csv', index=False)
print('Submission saved.')
```