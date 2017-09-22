import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# load data
def get_combined_data():
    train_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    targets = train_set['label']
    train_set.drop(["label"], axis=1, inplace=True)
    combined = train_set.append(test_set)
    combined.reset_index(inplace=True)
    combined.drop(["index"], axis=1, inplace=True)
    return combined, targets

combined, targets = get_combined_data()
train_set = combined.head(42000)
test_set = combined.iloc[42000:]

run_gridsearch = False
if run_gridsearch:
    param_grid = {
        'C': [1, 3, 5, 10, 20, 50, 100],
        'kernel': ['rbf', 'lin']
    }
    svc = SVC()
    cross_validation = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(svc,
                               scoring='accuracy',
                               param_grid=param_grid,
                               cv=cross_validation)
    grid_search.fit(train_set, targets)
    model = grid_search
    print 'Best score: {}'.format(model.best_score_)
    print 'Best params: {}'.format(model.best_params_)
else:
    params = {
        'C': 1.,
        'kernel': 'rbf'
    }
    svc = SVC(**params)
    svc.fit(train_set, targets)
    model = svc


def compute_score(clf, train_set, targets, scoring='accuracy'):
    xval = cross_val_score(clf, train_set, targets, cv=10, scoring=scoring)
    return np.mean(xval)

print 'Finally predict accuracy: ', compute_score(model, train_set, targets)

submission = pd.DataFrame({
    "ImageId": xrange(1, 28001),
    "Label": model.predict(test_set).astype(int)
})
submission.to_csv('predict.csv', index=False)
