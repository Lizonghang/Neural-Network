"""
Thanks to Ahmed Besbes' work in titanic kaggle challenge.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.grid_search import GridSearchCV

# I - Exploratory data analysis
"""
# loading the training set
data = pd.read_csv('train.csv')

# replace the null values with the median age.
data['Age'].fillna(data['Age'].median(), inplace=True)

# visualize survival based on the gender
survived_sex = data[data['Survived'] == 1]['Sex'].value_counts()
dead_sex = data[data['Survived'] == 0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex, dead_sex])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.show()

# visualize survival based on the age
figure = plt.figure(figsize=(15, 8))
plt.hist([data[data['Survived'] == 1]['Age'], data[data['Survived'] == 0]['Age']],
         stacked=True, bins=30, label=['Survived', 'Dead'], color=['g', 'r'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()

# visualize survival based on the fare
figure = plt.figure(figsize=(15, 8))
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']],
         stacked=True, color=['g', 'r'], bins=30, label=['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

# combine the age, the fare and the survival on a chart
plt.figure(figsize=(15, 8))
ax = plt.subplot()
ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'],
           c='green', s=20)
ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'],
           c='red', s=20)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('Survived', 'Dead'), scatterpoints=1, loc='upper right', fontsize=15)
plt.show()

# show the ticket fare correlates with class
ax = plt.subplot()
ax.set_ylabel('Average Fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar', ax=ax)
plt.show()

# show how the embarkation site affects the survival
# in fact, there seems to be no distinct correlation here
survived_embark = data[data['Survived'] == 1]['Embarked'].value_counts()
dead_embark = data[data['Survived'] == 0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark, dead_embark])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.show()
"""


# II - Feature engineering

# conbine the training set and the test set together, it's useful when
# the test set appears to have a feature that doesn't exist in the
# training set.
def get_combined_data():
    train_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    targets = train_set['Survived']
    train_set.drop(['Survived'], axis=1, inplace=True)
    print 'Origin shape of train set: ', np.shape(train_set)

    combined = train_set.append(test_set)
    combined.reset_index(inplace=True)
    combined.drop(['index'], axis=1, inplace=True)
    return combined, targets

combined, targets = get_combined_data()


# extracting the passenger titles
def parse_titles(pd):
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    name_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }
    combined['Title'] = combined['Title'].map(name_dictionary)
    return combined

combined = parse_titles(combined)

# processing the ages
# group the dataset by sex, title and passenger class and for each subset compute the median age.
# to avoid data leakage from the test set, perform these operations separately on the train set and test set.
grouped_train = combined.head(891).groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()

grouped_test = combined.iloc[891:].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_test = grouped_test.median()


# fills in the missing age in combined based on grouped median value

def process_age(combined):
    def fill_ages(row, grouped_median):
        return grouped_median.loc[row['Sex'], row['Pclass'], row['Title']]['Age']
    combined.head(891)['Age'] = combined.head(891).apply(lambda r: fill_ages(r, grouped_median_train) if np.isnan(r['Age']) else r['Age'], axis=1)
    combined.iloc[891:]['Age'] = combined.iloc[891:].apply(lambda r: fill_ages(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis=1)
    return combined

combined = process_age(combined)


# process the names
def process_names(combined):
    combined.drop(['Name'], axis=1, inplace=True)
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)
    combined.drop(['Title'], axis=1, inplace=True)
    return combined

combined = process_names(combined)


# processing Fare
def process_fares(combined):
    # there's one missing fare value - replacing with the mean
    combined.head(891)['Fare'].fillna(combined.head(891)['Fare'].mean(), inplace=True)
    combined.iloc[891:]['Fare'].fillna(combined.iloc[891:]['Fare'].mean(), inplace=True)
    return combined

combined = process_fares(combined)


# processing Embarked
def process_embarked(combined):
    # two missing embarked values, filling with the most frequent one "S"
    combined.head(891)['Embarked'].fillna('S', inplace=True)
    combined.iloc[891:]['Embarked'].fillna('S', inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop(['Embarked'], axis=1, inplace=True)
    return combined

combined = process_embarked(combined)


# processing Cabin
def process_cabin(combined):
    # replacing missing cabins with U for unknown
    combined['Cabin'].fillna('U', inplace=True)
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    # dummy encoding
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)
    combined.drop(['Cabin'], axis=1, inplace=True)
    return combined

combined = process_cabin(combined)


# processing Sex
def process_sex(combined):
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})
    return combined

combined = process_sex(combined)


# processing Pclass
def process_pclass(combined):
    # dummy encoding
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix='Pclass')
    combined = pd.concat([combined, pclass_dummies], axis=1)
    combined.drop(['Pclass'], axis=1, inplace=True)
    return combined

combined = process_pclass(combined)


# Processing Ticket
def process_ticket(combined):
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def clean_ticket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = filter(lambda t: not t.isdigit(), ticket)
        if len(ticket) > 0: return ticket[0]
        else: return 'XXX'
    combined['Ticket'] = combined['Ticket'].map(clean_ticket)

    # dummy encoding
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop(['Ticket'], axis=1, inplace=True)
    return combined

combined = process_ticket(combined)


# processing Family
def process_family(combined):
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    # introducing other features based on the family size
    combined['Single'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

    combined.drop(['Parch', 'SibSp'], axis=1, inplace=True)
    return combined

combined = process_family(combined)

# remove the PassengerId variable. It conveys no information to the prediction task.
combined.drop(['PassengerId'], axis=1, inplace=True)

# III - Modeling


# recovering the train set and the test set from the combined dataset.
def recover_train_test_target(combined):
    train_set = combined.head(891)
    test_set = combined.iloc[891:]
    return train_set, test_set

train_set, test_set = recover_train_test_target(combined)
print 'Shape of train set after Sparse Coding: ', np.shape(train_set)

# Feature selection
# When feature engineering is done, we usually tend to decrease the dimensionality
# by selecting the right number of features that capture the essential.
# Feature selection comes with many benefits:
# 1. decreases redundancy among the data.
# 2. speed up the training process
# 3. reduces overfitting
# Tree-based estimators can be used to compute feature importances, which in turn can
# be used to discard irrelevant features.
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train_set, targets)

# have a look at the importance of each feature.
feature = pd.DataFrame()
feature['feature'] = train_set.columns
feature['importance'] = clf.feature_importances_
feature.sort_values(by=['importance'], ascending=True, inplace=True)
feature.set_index('feature', inplace=True)
feature.plot(kind='barh', figsize=(20, 10))
plt.show()

# transform train set and test set in a more compact datasets.
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train_set)
test_reduced = model.transform(test_set)
print 'Shape of train set after Feature Selection: ', np.shape(train_reduced)

# Hyperparameters tuning
# train using a Random Forest Classifier
run_gridsearch = False
if run_gridsearch:
    param_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [1., 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False]
    }
    random_forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)
    grid_search = GridSearchCV(random_forest,
                               scoring='accuracy',
                               param_grid=param_grid,
                               cv=cross_validation)
    grid_search.fit(train_set, targets)
    model = grid_search
    print 'Best score: {}'.format(grid_search.best_score_)
    print 'Best params: {}'.format(grid_search.best_params_)
else:
    params = {
        'max_depth': 6,
        'n_estimators': 45,
        'max_features': 'auto',
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'bootstrap': True
    }
    random_forest = RandomForestClassifier(**params)
    random_forest.fit(train_set, targets)
    model = random_forest


# define a scoring function
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)

print 'Finally predict accuracy: ', compute_score(model, train_set, targets, scoring='accuracy')

# generate submission file
submission = pd.DataFrame({
    "PassengerId": pd.read_csv('test.csv')['PassengerId'],
    "Survived": model.predict(test_set).astype(int)
})
submission.to_csv('predict.csv', index=False)
