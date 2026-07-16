import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = 'input/train.csv'
TEST_PATH = 'input/test.csv'
SUBMIT_PATH = 'output/submission.csv'
N_FOLDS = 5
RANDOM_STATE = 42

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub_ids = test['Id']

outlier_ids = [524, 1299]
train = train[~train['Id'].isin(outlier_ids)].reset_index(drop=True)

y = np.log1p(train['SalePrice'])

train.drop(columns='Id', inplace=True)
test.drop(columns='Id', inplace=True)

neighborhood_median = train.groupby('Neighborhood')['SalePrice'].median()
train['Neighborhood_median'] = train['Neighborhood'].map(neighborhood_median)
test['Neighborhood_median'] = test['Neighborhood'].map(neighborhood_median)

cat_cols = train.select_dtypes('object').columns.tolist()
num_cols = train.select_dtypes(exclude='object').columns.tolist()
num_cols = [c for c in num_cols if c != 'SalePrice']

high_missing = [c for c in num_cols if train[c].isnull().mean() > 0.5]
num_cols = [c for c in num_cols if c not in high_missing]

ord_maps = {
    'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'PoolQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtExposure': {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3},
    'BsmtFinType1': {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5},
    'BsmtFinType2': {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5},
    'Functional': {'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},
    'Fence': {'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
}

for c in cat_cols:
    if c in ord_maps:
        train[c] = train[c].map(ord_maps[c]).fillna(0)
        test[c] = test[c].map(ord_maps[c]).fillna(0)

nominal_cols = [c for c in cat_cols if c not in ord_maps]

old_cols = set(train.columns)
for c in nominal_cols:
    train[c] = train[c].fillna('__MISSING__').astype(str)
    test[c] = test[c].fillna('__MISSING__').astype(str)

combined = pd.concat([train, test], axis=0)
combined = pd.get_dummies(combined, columns=nominal_cols, drop_first=True)
dummy_cols = [c for c in combined.columns if c not in old_cols or c in nominal_cols]
train = combined.iloc[:len(train)].reset_index(drop=True)
test = combined.iloc[len(train):].reset_index(drop=True)
cat_cols = [c for c in cat_cols if c in ord_maps]

for c in num_cols:
    med = train[c].median()
    train[c] = train[c].fillna(med)
    test[c] = test[c].fillna(med)

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF']  = test['TotalBsmtSF']  + test['1stFlrSF']  + test['2ndFlrSF']
train['HouseAge'] = train['YrSold'] - train['YearBuilt']
test['HouseAge']  = test['YrSold']  - test['YearBuilt']
train['QualCond'] = train['OverallQual'] * train['OverallCond']
test['QualCond']  = test['OverallQual']  * test['OverallCond']
num_cols += ['TotalSF', 'HouseAge', 'QualCond']

all_features = cat_cols + num_cols + dummy_cols
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
        depth=5,
        learning_rate=0.033,
        l2_leaf_reg=6.6,
        subsample=0.66,
        colsample_bylevel=0.81,
        min_data_in_leaf=3,
        random_strength=6.9,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=100)
    test_preds += np.expm1(model.predict(X_test)) / N_FOLDS
    oof_preds[va_idx] = np.expm1(model.predict(X_va))

rmse = np.sqrt(np.mean((np.log1p(oof_preds) - np.log1p(train['SalePrice']))**2))
print(f'CV RMSE (log): {rmse:.6f}')

sub = pd.DataFrame({'Id': sub_ids, 'SalePrice': test_preds})
sub.to_csv(SUBMIT_PATH, index=False)
print(f'Submission saved to {SUBMIT_PATH}')
