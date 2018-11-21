
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.svm import SVC #support vector Machine
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
# Any results you write to the current directory are saved as output.

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
PassengerId = test['PassengerId']

full_data = [train, test]

train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

train['Has_Cabin'] = train['Cabin'].apply(lambda x : 0 if type(x) == float else 1)

test['Has_Cabin'] = test['Cabin'].apply(lambda x : 0 if type(x) == float else 1)

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'],4)

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std,age_avg + age_std , size = age_null_count)
    dataset.loc[np.isnan(dataset['Age']),'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'],5)

def get_title(name):
    title_search = re.search('([A-Za-z]+)\.',name)
    if title_search :
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev',
                                                 'Sir','Jonkheer','Dona'],'Rare')

    dataset['Title'] = dataset['Title'].replace('Mile','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')

for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    title_mapping = {'Mr' : 1, 'Miss': 2, 'Mrs':3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454) , 'Fare'] =1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] =3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis= 1)
train = train.drop(['CategoricalAge','CategoricalFare'], axis= 1)
test = test.drop(drop_elements, axis = 1)

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson correlation of Features ', y = 1.05,size = 15)
sns.heatmap(train.astype(float).corr(), linewidths= 0.1, vmax= 1.0,
            square= True, cmap= colormap, linecolor= 'white', annot= True)

g = sns.pairplot(train[['Survived', 'Pclass', 'Sex', 'Age', 'Parch','Fare', 'Embarked', 'FamilySize',
                 'Title']], hue= 'Survived',palette= 'seismic', size= 1.2, diag_kind= 'kde',
                 diag_kws=dict(shade = True), plot_kws=dict(s = 10))
g.set(xticklabels = [])

"""


"""
# ntrain = train.shape[0]
# ntest = test.shape[0]
# kf = KFold( n_splits= 5 ,random_state= 0)
# def get_oof(clf, x_train, y_train, x_test):
#     off_train = np.zeros((ntrain))
#     off_test = np.zeros((ntest))
#     off_test_skf = np.empty((5,ntest))
#
#     for i , (train_index, test_index) in  enumerate(kf.split(x_train)):
#         x_tr = x_train[train_index]
#         y_tr = y_train[train_index]
#         x_te = x_train[test_index]
#         clf.train(x_tr,y_tr)
#
#         off_train[test_index] = clf.predict(x_te)
#         off_test_skf[i,:] = clf.predict(x_test)
#
#     off_test[:] = off_test_skf.mean(axis= 0)
#     return off_train.reshape(-1,1), off_test.reshape(-1,1)

""""
model 
"""
cv_train,cv_test=train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])
cv_train_X=cv_train[cv_train.columns[1:]].values
cv_train_Y=cv_train[cv_train.columns[:1]].values.ravel()
cv_test_X=cv_test[cv_test.columns[1:]].values
cv_test_Y=cv_test[cv_test.columns[:1]].values
X=train[train.columns[1:]].values
X_test=test[test.columns[:]].values
Y=train['Survived']


def Model_pre(clf,cv_train_X,cv_train_Y,X,Y,model_name):
    print("-------------------%s---------------------"%model_name)
    model = clf
    model.fit(cv_train_X,cv_train_Y)
    prediction_lr=model.predict(cv_test_X)
    print('The accuracy of the %s is'%model_name,round(accuracy_score(prediction_lr,cv_test_Y)*100,2))
    print("in the initial data set ,not val prediction!!!!!")
    result=cross_val_score(model,X,Y,cv=5,scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (result.mean(), result.std() * 2))
    y_pred = cross_val_predict(model,X,Y,cv=5)
   #  plt.figure()
   # #print("cross_val_predict is :",y_pred.shape )
   #  sns.heatmap(confusion_matrix(Y,y_pred),annot=True,fmt='3.0f',cmap="summer")
   #  plt.title('%s Confusion_matrix'%model_name, y=1.05, size=15)
   #  plt.show()
    return result
model = LogisticRegression()
result_lr = Model_pre(model,cv_train_X,cv_train_Y,X,Y,"LogisticRegression")

model = RandomForestClassifier(criterion="entropy",max_depth=6, n_estimators=50)
result_rf = Model_pre(model,cv_train_X,cv_train_Y,X,Y,"RandomForestClassifier")
model = SVC()

result_svm = Model_pre(model,cv_train_X,cv_train_Y,X,Y,"SVC")

model = KNeighborsClassifier()
result_knn = Model_pre(model,cv_train_X,cv_train_Y,X,Y,"KNeighborsClassifier")

model= GaussianNB()
result_gbc = Model_pre(model,cv_train_X,cv_train_Y,X,Y,"GaussianNB")

model= DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')

result_tree = Model_pre(model,cv_train_X,cv_train_Y,X,Y,"DecisionTreeClassifier")

model= AdaBoostClassifier()
result_adb = Model_pre(model,cv_train_X,cv_train_Y,X,Y,"AdaBoostClassifier")


model= GradientBoostingClassifier( learning_rate = 0.05, loss ='exponential',  max_depth = 3, n_estimators =200)
result_gnb =  Model_pre(model,cv_train_X,cv_train_Y,X,Y,"GradientBoostingClassifier")

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', ' GradientBoostingClassifier', 'AdaBoostClassifier',
              'GaussianNB',
              'Decision Tree'],
    'Score': [result_svm.mean(), result_knn.mean(), result_lr.mean(),
              result_rf.mean(), result_gnb.mean(), result_adb.mean(),
              result_gbc.mean(), result_tree.mean()]})
models.sort_values(by='Score',ascending=False)
print(models.sort_values(by='Score',ascending=False))

model= DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')
model.fit(X,Y)
Y_pred_re =model.predict(X_test)
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': Y_pred_re })
StackingSubmission.to_csv("RF.csv", index=False)



""""
other ways to try

"""

# Random Forest

# param_grid = { 'n_estimators': [10, 50, 100, 300], #default=10
#             'criterion': ['gini', 'entropy'], #default=”gini”
#             'max_depth': [2, 4, 6, 8, 10, None],
#
#              }
#
# tune_model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = 5)
# tune_model.fit(X, Y)
#
# print('AFTER DT Parameters: ', tune_model.best_params_)
# #print(tune_model.cv_results_['mean_train_score'])
# print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
# #print(tune_model.cv_results_['mean_test_score'])
# print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
# print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
# print('-'*10)


##
#
# def Get_train_test_label(clf,X,Y):
#     model = clf
#     model.fit(X,Y)
#     result_train=cross_val_predict(model,X,Y,cv=5)
#     result_test = model.predict(X_test)
#     return  result_train, result_test
#
# model= DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=0,
#             splitter='best')
#
# DT_train,DT_test =  Get_train_test_label(model,X,Y)
#
#
# model= AdaBoostClassifier()
# AdaBoostClassifier_train,AdaBoostClassifier_test =  Get_train_test_label(model,X,Y)
#
# model = SVC()
# svc_train,svc_test =  Get_train_test_label(model,X,Y)
#
# model= GradientBoostingClassifier( learning_rate = 0.05, loss ='exponential',  max_depth = 3, n_estimators =200)
#
# Gb_train,Gb_test =  Get_train_test_label(model,X,Y)
#
# model = RandomForestClassifier(criterion="entropy",max_depth=6, n_estimators=50)
# RF_train,RF_test =  Get_train_test_label(model,X,Y)
# x_train = np.concatenate(( DT_train.reshape((-1,1)), AdaBoostClassifier_train.reshape((-1,1)), svc_train.reshape((-1,1)), Gb_train.reshape((-1,1)), RF_train.reshape((-1,1))), axis=1)
# x_test = np.concatenate(( DT_test.reshape((-1,1)), AdaBoostClassifier_test.reshape((-1,1)), Gb_test.reshape((-1,1)), Gb_test.reshape((-1,1)), RF_test.reshape((-1,1))), axis=1)
#
# x_test_sum = x_test.sum(axis= 1)
#
# for i , v in enumerate(x_test_sum):
#     if v < 2 :
#         x_test_sum[i] = 0
#     else:
#         x_test_sum[i] = 1
# # map({'female': 0, 'male': 1}).astype(int)
# #
# StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
#                             'Survived':  x_test_sum})
# StackingSubmission.to_csv("best1.csv", index=False)
#
