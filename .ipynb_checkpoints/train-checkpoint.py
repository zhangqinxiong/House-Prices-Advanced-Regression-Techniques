import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
SUBMIT_PATH = 'submission.csv'
N_FOLDS = 5
RANDOM_STATE = 42

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub_ids = test['Id']
y = np.log1p(train['SalePrice'])

train.drop(columns='Id', inplace=True)
test.drop(columns='Id', inplace=True)

cat_cols = train.select_dtypes('object').columns.tolist()
num_cols = train.select_dtypes(exclude='object').columns.tolist()
num_cols = [c for c in num_cols if c != 'SalePrice']

high_missing = [c for c in num_cols if train[c].isnull().mean() > 0.5]
num_cols = [c for c in num_cols if c not in high_missing]

for c in cat_cols:
    le = LabelEncoder()
    full = pd.concat([train[c], test[c]]).fillna('__MISSING__').astype(str)
    le.fit(full)
    train[c] = le.transform(train[c].fillna('__MISSING__').astype(str))
    test[c] = le.transform(test[c].fillna('__MISSING__').astype(str))

for c in num_cols:
    med = train[c].median()
    train[c] = train[c].fillna(med)
    test[c] = test[c].fillna(med)

all_features = cat_cols + num_cols
X = train[all_features]
X_test = test[all_features]

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = CatBoostRegressor(
        task_type='CPU',
        random_seed=RANDOM_STATE,
        verbose=100,
        early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=100)
    test_preds += np.expm1(model.predict(X_test)) / N_FOLDS
    oof_preds[va_idx] = np.expm1(model.predict(X_va))

rmse = np.sqrt(np.mean((np.log1p(oof_preds) - np.log1p(train['SalePrice']))**2))
print(f'CV RMSE (log): {rmse:.6f}')

sub = pd.DataFrame({'Id': sub_ids, 'SalePrice': test_preds})
sub.to_csv(SUBMIT_PATH, index=False)
print(f'Submission saved to {SUBMIT_PATH}')
