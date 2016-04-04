import pandas
import numpy
import sklearn
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import maxabs_scale, minmax_scale
from sklearn.grid_search import GridSearchCV
from linear_discriminant_fun import *
import seaborn

pandas.set_option('expand_frame_repr', False)
pandas.set_option('display.max_rows', 1000)

train_data = pandas.read_csv('train_preprocessed.csv')
test_data = pandas.read_csv('test_preprocessed.csv')

## insert survived column to test_data
test_data.insert(1, 'Survived', 0)
#############  END preprocess ##############

#############  model training ##############

### ########## decision tree
### 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Has_Family'
labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Has_Family'] #'Embarked', 
train_split, test_split = train_test_split(train_data, train_size=0.9)

### ### ### grid search
### ### dt_clf = DecisionTreeClassifier()
### ### parameters_grid = [{'criterion':['entropy', 'gini'], 'max_depth':range(1,8), 'min_samples_split':numpy.arange(5,50,2)}]
### ### grid_search = GridSearchCV(dt_clf, parameters_grid, cv=10, n_jobs=2)
### ### grid_search.fit(train_data[labels], train_data['Survived'])
### ### print grid_search.best_params_, grid_search.best_score_
### 
### dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=13)
### cv_score = cross_val_score(dt_clf, 
###         train_split[labels],
###         train_split['Survived'], cv=10)
### print 'decision tree cv_score=', cv_score, ' mean=', numpy.mean(cv_score)
### dt_clf.fit(train_split[labels], train_split['Survived'])
### print 'dt train_split score=', dt_clf.score(train_split[labels], train_split['Survived'])
### print 'dt test_split score=', dt_clf.score(test_split[labels], test_split['Survived'])
### 
### dt_clf.fit(train_data[labels], train_data['Survived'])
### print 'dt train score=', dt_clf.score(train_data[labels], train_data['Survived'])
### test_data['Survived'] = dt_clf.predict(test_data[labels])
### test_data.to_csv('norm_dt_class.csv', columns=['PassengerId',  'Survived'], index=False)
### sklearn.tree.export_graphviz(dt_clf.tree_, out_file='dt_clf_tree.dot')

### parameters_grid = [{'criterion':['entropy', 'gini'], 'max_depth':range(1,8), 'min_samples_split':numpy.arange(5,50,2)}]
### grid_search = GridSearchCV(dt_clf, parameters_grid, cv=10, n_jobs=2)
### grid_search.fit(train_data[labels], train_data['Survived'])
### print grid_search.best_params_, grid_search.best_score_

rf_clf = RandomForestClassifier()
parameters_grid = [{'criterion':['entropy', 'gini'], 'max_depth':range(1,10), 'min_samples_split':numpy.arange(1,50,1), 'n_estimators':numpy.arange(10,500,50)}]
grid_search = GridSearchCV(rf_clf, parameters_grid, cv=5, n_jobs=4)
grid_search.fit(train_data[labels], train_data['Survived'])
print grid_search.best_params_, grid_search.best_score_

## print 'randome forest, cv score=', cross_val_score(rf_clf, train_data[labels], train_data['Survived'], cv=10)
## rf_clf.fit(train_data[labels], train_data['Survived'])
## test_data['Survived'] = rf_clf.predict(test_data[labels])
## test_data.to_csv('norm_rf_class.csv', columns=['PassengerId',  'Survived'], index=False)

### TODO look deep into here
### # this branch has 85, 54 partition
### # train_data[(train_data['Sex']==0) & (train_data['Fare']>=26.2687) & (train_data['SibSp']<2.5)]
### # this branch has 48, 69 partition
### # train_data[(train_data['Sex']==1) & (train_data['Fare']<=23.35) & (train_data['Pclass']>2.5)]

#### ### write these mis-classified data into files
#### mis_classified_group_1 = train_data[
####    ( (train_data['Sex']==0) & (train_data['Fare']>=26.2687) & (train_data['SibSp']<2.5) ) ]
#### mis_classified_group_2 = train_data[
####    ( (train_data['Sex']==1) & (train_data['Fare']<=23.35) & (train_data['Pclass']>2.5) ) ]
#### mis_classified_group = pandas.concat([mis_classified_group_1, mis_classified_group_2])
#### normalized_train_data = minmax_scale(mis_classified_group[labels].values)
#### 
#### ### group_dict = {'group':mis_classified_group, 'group1':mis_classified_group_1, 'group2':mis_classified_group_2}
#### ### 
#### ### for k, mis_group in group_dict.items():
#### ###     normalized_train_data = maxabs_scale(mis_group[labels].values)
#### ###     #{'kernel': ['sigmoid'], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100], 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'coef0':[1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
#### ###     parameters_grid = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100], 'C': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]}]
#### ###     clf = SVC()
#### ###     grid_search = GridSearchCV(clf, parameters_grid, cv=10, n_jobs=2)
#### ###     grid_search.fit(normalized_train_data, mis_group['Survived'])
#### ###     print grid_search.best_params_, grid_search.best_score_
####      
#### svm_clf = SVC(C=0.6, kernel='rbf', gamma=5)
#### svm_clf.fit(normalized_train_data, mis_classified_group['Survived'])
#### print 'svm on mis classified group=', svm_clf.score(normalized_train_data, mis_classified_group['Survived'])
#### #test_data.to_csv('dt_svm_class.csv', columns=['PassengerId',  'Survived'], index=False)
#### 
#### knn_clf = KNeighborsClassifier(n_neighbors=1, p=1)
#### knn_clf.fit(normalized_train_data, mis_classified_group['Survived'])
#### print 'KNN on mis classified group=', knn_clf.score(normalized_train_data, mis_classified_group['Survived'])
#### 
#### lr_clf = LogisticRegression()
#### lr_clf.fit(normalized_train_data, mis_classified_group['Survived'])
#### print 'LR on mis classified group=', lr_clf.score(normalized_train_data, mis_classified_group['Survived'])

### mis_classified_group.to_csv('dt_mis_classified.csv')
### mis_classified_group_1.to_csv('dt_mis_classified_1.csv')
### mis_classified_group_2.to_csv('dt_mis_classified_2.csv')

### ## Basic feature factor plot
### for group_name, train_data in group_dict.items():
###     seaborn.factorplot(x='Embarked', y='Survived', hue='Sex', kind="bar", data=train_data)
###     seaborn.plt.savefig(group_name+'-survived-embark-sex.pdf')
###     seaborn.plt.clf()
###     seaborn.countplot(x='Embarked', hue='Sex', data=train_data)
###     seaborn.plt.savefig(group_name+'-survived-count-embark-sex.pdf')
###     seaborn.plt.clf()
### 
###     seaborn.factorplot(x='Pclass', y='Survived', hue='Sex', kind="bar", data=train_data)
###     seaborn.plt.savefig(group_name+'-survived-pclass-sex.pdf')
###     seaborn.plt.clf()
###     seaborn.countplot(x='Pclass', hue='Sex', data=train_data)
###     seaborn.plt.savefig(group_name+'-survived-count-pclass-sex.pdf')
###     seaborn.plt.clf()
### 
###     seaborn.factorplot(x='Has_Family', y='Survived', hue='Sex', kind="bar", data=train_data)
###     seaborn.plt.savefig(group_name+'-survived-has-family-sex.pdf')
###     seaborn.plt.clf()
###     seaborn.countplot(x='Has_Family', hue='Sex', data=train_data)
###     seaborn.plt.savefig(group_name+'-survived-count-has-family-sex.pdf')
###     seaborn.plt.clf()
