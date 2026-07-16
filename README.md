# House Prices - Advanced Regression Techniques

[Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Predict sales prices of residential homes in Ames, Iowa using 79 explanatory variables.

## Approach

- **Model**: CatBoostRegressor with 5-fold cross validation
- **Preprocessing**: Ordinal encoding, one-hot encoding, missing value imputation
- **Feature Engineering**: Neighborhood median price, total square footage, house age, quality*condition interaction
- **Outlier Removal**: Removes IDs 524 and 1299

## Project Structure

```
├── input/
│   ├── train.csv              # Training data (1460 rows, 81 columns)
│   ├── test.csv                # Test data (1459 rows, 80 columns)
│   └── sample_submission.csv   # Sample submission format
├── output/
│   └── submission.csv          # Generated predictions
├── train.py                    # Main training script
└── README.md
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- catboost

## Usage

```bash
pip install pandas numpy scikit-learn catboost
python train.py
```

The script outputs cross-validation RMSE and generates `output/submission.csv`.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Folds | 5 |
| Depth | 5 |
| Learning Rate | 0.033 |
| L2 Reg | 6.6 |
| Subsample | 0.66 |
| Colsample | 0.81 |
| Early Stopping | 50 rounds |
