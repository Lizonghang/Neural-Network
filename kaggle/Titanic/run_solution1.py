import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# load data and drop unnecessary columns
train_df = pd.read_csv("train.csv").drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_df = pd.read_csv("test.csv").drop(['Name', 'Ticket'], axis=1)

# Embarked
"""
# only in train_df, fill the missing values with the most occurred value 'S'
train_df["Embarked"].fillna("S", inplace=True)

embarked_dummies_train = pd.get_dummies(train_df["Embarked"])
embarked_dummies_train.columns = ["S", "C", "Q"]
# embarked_dummies_train.drop(["S"], axis=1, inplace=True)

embarked_dummies_test = pd.get_dummies(test_df["Embarked"])
embarked_dummies_test.columns = ["S", "C", "Q"]
# embarked_dummies_test.drop(["S"], axis=1, inplace=True)

train_df.drop(["Embarked"], axis=1, inplace=True)
test_df.drop(["Embarked"], axis=1, inplace=True)

train_df = train_df.join(embarked_dummies_train)
test_df = test_df.join(embarked_dummies_test)
"""
train_df.drop(["Embarked"], axis=1, inplace=True)
test_df.drop(["Embarked"], axis=1, inplace=True)

# Fare

# only in test_df, fill the missing values with the median value of "Fare"
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
train_df["Fare"] = train_df["Fare"].astype(int)
test_df["Fare"] = test_df["Fare"].astype(int)

# Age

# get average, std, and number of NaN values in train_df
average_age_train = train_df["Age"].mean()
std_age_train = train_df["Age"].std()
nan_age_train = train_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test = test_df["Age"].mean()
std_age_test = test_df["Age"].std()
nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand1 = np.random.randint(average_age_train - std_age_train,
                          average_age_train + std_age_train,
                          size=nan_age_train)
rand2 = np.random.randint(average_age_test - std_age_test,
                          average_age_test + std_age_test,
                          size=nan_age_test)

# fill NaN values in Age column with random values generated
train_df["Age"][np.isnan(train_df["Age"])] = rand1
test_df["Age"][np.isnan(test_df["Age"])] = rand2

# convert from float to int
train_df["Age"] = train_df["Age"].astype(int)
test_df["Age"] = test_df["Age"].astype(int)

# Cabin
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
train_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)

# Family
"""
# Instead of having two columns Parch & SibSp,
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
train_df["Family"] = train_df["Parch"] + train_df["SibSp"]
train_df["Family"].loc[train_df["Family"] > 0] = 1
train_df["Family"].loc[train_df["Family"] == 0] = 0

test_df["Family"] = test_df["Parch"] + test_df["SibSp"]
test_df["Family"].loc[test_df["Family"] > 0] = 1
test_df["Family"].loc[test_df["Family"] == 0] = 0

# drop Parch & SibSp
train_df.drop(["SibSp", "Parch"], axis=1, inplace=True)
test_df.drop(["SibSp", "Parch"], axis=1, inplace=True)
"""
train_df.drop(["Parch", "SibSp"], axis=1, inplace=True)
test_df.drop(["Parch", "SibSp"], axis=1, inplace=True)


# Sex
# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age, sex = passenger
    return "child" if age < 16 else sex

train_df["Person"] = train_df[["Age", "Sex"]].apply(get_person, axis=1)
test_df["Person"] = test_df[["Age", "Sex"]].apply(get_person, axis=1)

# Drop Sex column
train_df.drop(["Sex"], axis=1, inplace=True)
test_df.drop(["Sex"], axis=1, inplace=True)

# create dummy variables for Person column
person_dummies_train = pd.get_dummies(train_df["Person"])
person_dummies_train.columns = ["Child", "Female", "Male"]
# person_dummies_train.drop(["Male"], axis=1, inplace=True)

person_dummies_test = pd.get_dummies(test_df["Person"])
person_dummies_test.columns = ["Child", "Female", "Male"]
# person_dummies_test.drop(["Male"], axis=1, inplace=True)

train_df = train_df.join(person_dummies_train)
test_df = test_df.join(person_dummies_test)

train_df.drop(["Person"], axis=1, inplace=True)
test_df.drop(["Person"], axis=1, inplace=True)

# Pclass
# create dummy variables for Pclass column
pclass_dummies_train = pd.get_dummies(train_df["Pclass"])
pclass_dummies_train.columns = ["Class1", "Class2", "Class3"]
# pclass_dummies_train.drop(["Class3"], axis=1, inplace=True)

pclass_dummies_test = pd.get_dummies(test_df["Pclass"])
pclass_dummies_test.columns = ["Class1", "Class2", "Class3"]
# pclass_dummies_test.drop(["Class3"], axis=1, inplace=True)

train_df.drop(["Pclass"], axis=1, inplace=True)
test_df.drop(["Pclass"], axis=1, inplace=True)

train_df = train_df.join(pclass_dummies_train)
test_df = test_df.join(pclass_dummies_test)

# define training and testing sets
X_train = train_df.drop(["Survived"], axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop(["PassengerId"], axis=1).copy()

# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
logistic_regression_predict = logistic_regression.predict(X_test)
print 'Apply logistic regression on train set, precision: ', logistic_regression.score(X_train, Y_train)

# Support Vector Machines
svc = SVC(C=3., gamma=0.5)
svc.fit(X_train, Y_train)
svc_predict = svc.predict(X_test)
print 'Apply SVC on train set, precision: ', svc.score(X_train, Y_train)

# Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest_predict = random_forest.predict(X_test)
print 'Apply random forest on train set, precision: ', random_forest.score(X_train, Y_train)

# KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
knn_predict = knn.predict(X_test)
print 'Apply knn on train set, precision: ', knn.score(X_train, Y_train)

# Gaussian Naive Bayes
gaussianNB = GaussianNB()
gaussianNB.fit(X_train, Y_train)
gaussianNB_predict = gaussianNB.predict(X_test)
print 'Apply gaussian naive bayes on train set, precision: ', gaussianNB.score(X_train, Y_train)

all_predict = np.array([logistic_regression_predict,
                        svc_predict,
                        random_forest_predict,
                        knn_predict,
                        gaussianNB_predict])
Y_predict = all_predict.sum(axis=0) > (all_predict.shape[0] - 1) / 2.0
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_predict.astype(int)
})
submission.to_csv("predict.csv", index=False)
