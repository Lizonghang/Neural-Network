import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.grid_search import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('dataset/train.csv')
targets = df_train['SalePrice']
df_train.drop("SalePrice", axis=1, inplace=True)
df_test = pd.read_csv('dataset/test.csv')
combined = df_train.append(df_test)
combined.reset_index(inplace=True)
combined.drop(["index", "Id"], axis=1, inplace=True)

# dealing with missing data
total = combined.isnull().sum()
combined.drop(total[total > 1].index, axis=1, inplace=True)
combined['KitchenQual'].fillna('TA', inplace=True)
combined['Exterior1st'].fillna('VinylSd', inplace=True)
combined['Exterior2nd'].fillna('VinylSd', inplace=True)
combined['BsmtFinSF1'].fillna(0, inplace=True)
combined['BsmtFinSF2'].fillna(0, inplace=True)
combined['BsmtUnfSF'].fillna(250, inplace=True)
combined['TotalBsmtSF'].fillna(880, inplace=True)
combined['Electrical'].fillna('SBrkr', inplace=True)
combined['GarageCars'].fillna(2, inplace=True)
combined['GarageArea'].fillna(490, inplace=True)
combined['SaleType'].fillna('WD', inplace=True)

# Need normalize

combined = pd.get_dummies(combined)

df_train = combined.head(1460)
df_test = combined.iloc[1460:]

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(df_train, targets)
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(df_train)
test_reduced = model.transform(df_test)

# Use DNN Model