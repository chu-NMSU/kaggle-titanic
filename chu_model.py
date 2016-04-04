import pandas
import numpy
import sklearn
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import maxabs_scale
from sklearn.grid_search import GridSearchCV
from linear_discriminant_fun import *

pandas.set_option('expand_frame_repr', False)
pandas.set_option('display.max_rows', 1000)

train_data = pandas.read_csv('train_preprocessed.csv')
test_data = pandas.read_csv('test_preprocessed.csv')

## insert survived column to test_data
test_data.insert(1, 'Survived', 0)
#############  END preprocess ##############

#############  model training ##############

### ########## decision tree
labels = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Has_Family', 'Embarked_C', 'Embarked_Q', 'Embarked_S'] #['Pclass', 'Sex', 'Age', 'Embarked_C', 'Embarked_Q', 'Embarked_S',]

### grid search
## clf = DecisionTreeClassifier()
## parameters_grid = [{'criterion':['entropy', 'gini'], 'max_depth':range(1,10), 'min_samples_split':numpy.arange(5,50,2)}]
## grid_search = GridSearchCV(clf, parameters_grid, cv=10, n_jobs=2)
## grid_search.fit(train_data[labels], train_data['Survived'])
## print grid_search.best_params_, grid_search.best_score_

# {'min_samples_split': 15, 'criterion': 'gini', 'max_depth': 7} 0.833894500561
clf = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=15)
cv_score = cross_val_score(clf, 
        train_data[labels], 
        train_data['Survived'], cv=10)
print 'decision tree cv_score=', cv_score, ' mean=', numpy.mean(cv_score)
clf.fit(train_data[labels], train_data['Survived'])
print 'decision tree training score=', clf.score(train_data[labels], train_data['Survived'])
dt_predict = clf.predict(test_data[labels])
test_data['Survived'] = dt_predict
test_data.to_csv('dt_class.csv', columns=['PassengerId',  'Survived'], index=False)
### sklearn.tree.export_graphviz(clf.tree_, out_file='dt_clf_tree.dot')

### TODO look deep into here
### # this branch has 85, 54 partition
### # train_data[(train_data['Sex']==0) & (train_data['Fare']>=26.2687) & (train_data['SibSp']<2.5)]
### # this branch has 48, 69 partition
### # train_data[(train_data['Sex']==1) & (train_data['Fare']<=23.35) & (train_data['Pclass']>2.5)]

######## Navie Bayes.  NOTES: NB may does not make sense in this case
### clf = MultinomialNB(class_prior=[0.65, 0.35]) #GaussianNB() 
### labels =  ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch'] #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
### cv_score = cross_val_score(clf, 
###         train_data[labels], 
###         train_data['Survived'], cv=3)
### print 'navie bayes cv_score=', cv_score
### clf.fit(train_data[labels], train_data['Survived'])
### nb_predict = clf.predict(test_data[labels])
### test_data['Survived'] = nb_predict
### test_data.to_csv('nb_class.csv', columns=['PassengerId',  'Survived'], index=False)

###########linear regression
### reg = Lasso(alpha=0.001) #LinearRegression() 
labels =  ['Sex', 'Age', 'Fare', 'Has_Family', 'Embarked_S', 'Embarked_C', 'Embarked_Q'] #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
### kf = KFold(train_data.shape[0], n_folds=3)
threshold = 0.9
### print 'Linear Lassor regularization regression score='
### for train_index, test_index in kf:
###     reg.fit(train_data[labels].values[train_index,:], train_data['Survived'].values[train_index])
###     reg_predict = reg.predict(train_data[labels].values[test_index,:])
###     reg_predict[reg_predict>=threshold] = 1
###     reg_predict[reg_predict<threshold] = 0
###     reg_score = numpy.mean(reg_predict==train_data['Survived'].values[test_index])
###     print reg_score

### reg = LinearRegression()
### reg.fit(train_data[labels], train_data['Survived'])
### reg_predict = reg.predict(test_data[labels])
### reg_predict[reg_predict>=threshold] = 1
### reg_predict[reg_predict<threshold] = 0
### test_data['Survived'] = [int(i) for i in reg_predict]
### test_data.to_csv('lasso_class.csv', columns=['PassengerId',  'Survived'], index=False)

### ##############linear classifier, fisher classifier
### labels =  ['Sex', 'Age', 'Fare', 'Has_Family', 'Embarked_S', 'Embarked_C', 'Embarked_Q'] #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
### kf = KFold(train_data.shape[0], n_folds=3)
### print 'linear classifier'
### for train_index, test_index in kf:
###     tilde_W = LS_estimate(train_data[labels].values[train_index, :], 
###             train_data['Survived'].values[train_index], 2)
###     ls_predict = LS_predict(train_data[labels].values[test_index, :], tilde_W)
###     ls_score = numpy.mean(ls_predict==train_data['Survived'].values[test_index])
###     print ls_score
### 
### tilde_W = LS_estimate(train_data[labels].values, train_data['Survived'].values, 2)
### ls_predict = LS_predict(test_data[labels].values, tilde_W)
### test_data['Survived'] = ls_predict
### test_data.to_csv('ls_class.csv', columns=['PassengerId',  'Survived'], index=False)
### 
### print 'fisher linear classifier'
### for train_index, test_index in kf:
###     w, mean = fisher_linear_discriminant(train_data[labels].values[train_index, :], 
###             train_data['Survived'].values[train_index])
###     # print w 
###     fisher_predict = fisher_linear_predict(train_data[labels].values[test_index, :], w, mean, 0.000001)
###     fs_score = numpy.mean(fisher_predict==train_data['Survived'].values[test_index])
###     print fs_score
### 
### w, mean = fisher_linear_discriminant(train_data[labels].values, train_data['Survived'].values)
### fisher_predict = fisher_linear_predict(test_data[labels].values, w, mean, 0.000001)
### test_data['Survived'] = fisher_predict
### test_data.to_csv('fisher_class.csv', columns=['PassengerId',  'Survived'], index=False)

##############logistic regression
### clf = LogisticRegression(penalty='l2', dual=False, solver='liblinear', C=1.1)
### labels =  ['Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Has_Family', 'Embarked_S', 'Embarked_C', 'Embarked_Q'] #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
### cv_score = cross_val_score(clf, 
###         train_data[labels], 
###         train_data['Survived'], cv=3)
### print 'logistic regression cv_score=', cv_score

### clf.fit(train_data[labels], train_data['Survived'])
### lr_predict = clf.predict(test_data[labels])
### test_data['Survived'] = lr_predict
### test_data.to_csv('lr_class.csv', columns=['PassengerId',  'Survived'], index=False)

##############SVM
labels =  ['Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Has_Family', 'Embarked_S', 'Embarked_C', 'Embarked_Q'] #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
normalized_train_data = maxabs_scale(train_data[labels].values)
normalized_test_data = maxabs_scale(test_data[labels].values)

####### exhaustively CV grid search
### parameters_grid = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
###         {'kernel': ['sigmoid'], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100], 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'coef0':[1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50, 100]},
###     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
### clf = SVC()
### grid_search = GridSearchCV(clf, parameters_grid, cv=3, n_jobs=2)
### grid_search.fit(normalized_train_data, train_data['Survived'])
### print grid_search.best_params_, grid_search.best_score_
#### 
#### clf = SVC(C=5, kernel='rbf', gamma=1)
#### cv_score = cross_val_score(clf, 
####         normalized_train_data,
####         train_data['Survived'], cv=3)
#### print 'SVM cv_score=', cv_score
#### 
#### clf.fit(normalized_train_data, train_data['Survived'])
#### svm_predict = clf.predict(normalized_test_data)
#### test_data['Survived'] = svm_predict
#### test_data.to_csv('svm_class.csv', columns=['PassengerId',  'Survived'], index=False)

###############KNN
### labels =  ['Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Has_Family', 'Embarked_S', 'Embarked_C', 'Embarked_Q'] #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
### normalized_train_data = maxabs_scale(train_data[labels].values)
### normalized_test_data = maxabs_scale(test_data[labels].values)

### parameters_grid = [{'n_neighbors':range(1,50), 'p':range(1,10)}]
### clf =  KNeighborsClassifier()
### grid_search = GridSearchCV(clf, parameters_grid, cv=3, n_jobs=2)
### grid_search.fit(normalized_train_data, train_data['Survived'])
### print grid_search.best_params_, grid_search.best_score_

### clf =  KNeighborsClassifier(n_neighbors=4, p=1)
### clf.fit(normalized_train_data, train_data['Survived'])
### knn_predict = clf.predict(normalized_test_data)
### test_data['Survived'] = knn_predict
### test_data.to_csv('knn_class.csv', columns=['PassengerId',  'Survived'], index=False)

#################ensemble methods
labels =  ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Family', 'Embarked_S', 'Embarked_C', 'Embarked_Q'] #['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
normalized_train_data = maxabs_scale(train_data[labels].values)
normalized_test_data = maxabs_scale(test_data[labels].values)
### normalized_train_data = train_data[labels].values
### normalized_test_data = test_data[labels].values

############# bagging classifier
######## grid search parameters
base_svm = SVC(C=5, kernel='rbf', gamma=1)
base_knn = KNeighborsClassifier(n_neighbors=4, p=1)
base_dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=5)

bagging = BaggingClassifier()
### parameters_grid = [{'base_estimator':[base_svm], 'n_estimators':numpy.arange(5,100,5), 'max_samples':numpy.arange(0.1, 1, 0.1), 'max_features':numpy.linspace(0.5, 1.0, 10)}]
### grid_search = GridSearchCV(bagging, parameters_grid, cv=10, n_jobs=3)
### grid_search.fit(normalized_train_data, train_data['Survived'])
### print 'svm bagging grid search', grid_search.best_params_, grid_search.best_score_
### svm bagging grid search {'max_features': 0.77777777777777779, 'max_samples': 0.90000000000000002, 'base_estimator': SVC(C=5, cache_size=200, class_weight=None, coef0=0.0,
###   decision_function_shape=None, degree=3, gamma=1, kernel='rbf',
###     max_iter=-1, probability=False, random_state=None, shrinking=True,
###       tol=0.001, verbose=False), 'n_estimators': 5} 0.82379349046

### parameters_grid = [{'base_estimator':[base_knn], 'n_estimators':numpy.arange(5,100,5), 'max_samples':numpy.arange(0.1, 1, 0.1), 'max_features':numpy.linspace(0.5, 1.0, 10)}]
### grid_search = GridSearchCV(bagging, parameters_grid, cv=10, n_jobs=3)
### grid_search.fit(normalized_train_data, train_data['Survived'])
### print 'knn bagging grid search', grid_search.best_params_, grid_search.best_score_
###      knn bagging grid search {'max_features': 0.66666666666666663, 'max_samples': 0.59999999999999998, 'base_estimator': KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
###                     metric_params=None, n_jobs=1, n_neighbors=4, p=1,
###                                weights='uniform'), 'n_estimators': 20} 0.83164983165

## parameters_grid = [{'base_estimator':[base_dt], 'n_estimators':numpy.arange(5,100,5), 'max_samples':numpy.arange(0.1, 1, 0.1), 'max_features':numpy.linspace(0.5, 1.0, 10)}]
## grid_search = GridSearchCV(bagging, parameters_grid, cv=3, n_jobs=2)
## grid_search.fit(normalized_train_data, train_data['Survived'])
## print 'dt bagging grid search', grid_search.best_params_, grid_search.best_score_
## dt bagging grid search {'max_features': 0.72222222222222221, 'max_samples': 0.30000000000000004, 'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
##    max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
##    min_samples_split=5, min_weight_fraction_leaf=0.0,
##    presort=False, random_state=None, splitter='best'), 'n_estimators': 10} 0.827160493827

## bagging = BaggingClassifier(base_estimator=base_svm, n_estimators=50, max_samples=0.9, max_features=1.0)
## bagging = BaggingClassifier(base_estimator=base_knn, n_estimators=30, max_samples=0.6, max_features=0.94)
## bagging = BaggingClassifier(base_estimator=base_dt, n_estimators=10, max_samples=0.3, max_features=0.72)
## bagging_score = cross_val_score(bagging, normalized_train_data, train_data['Survived'], cv=3, n_jobs=2)
## print 'bagging cv=', bagging_score
## bagging.fit(normalized_train_data, train_data['Survived'])
## bagging_predict = bagging.predict(normalized_test_data)
## test_data['Survived'] = bagging_predict
## test_data.to_csv('bagging_svm_class.csv', columns=['PassengerId',  'Survived'], index=False)

########## random forest
### parameters_grid = [{'n_estimators':numpy.arange(5,100,5), 'criterion':['entropy'], 'max_depth':range(1,6), 'min_samples_split':numpy.arange(5, 50, 5), 'max_features':numpy.linspace(0.5, 1.0, 10)}]
### r_forest = RandomForestClassifier()
### grid_search = GridSearchCV(r_forest, parameters_grid, cv=10, n_jobs=2)
### grid_search.fit(normalized_train_data, train_data['Survived'])
### print 'random forest grid search', grid_search.best_params_, grid_search.best_score_
###      random forest grid search {'max_features': 0.83333333333333326, 'min_samples_split': 5, 'criterion': 'entropy', 'max_depth': 5, 'n_estimators': 30} 0.836139169473

## r_forest = RandomForestClassifier(max_features=0.72, min_samples_split=5, criterion='entropy', max_depth=5, n_estimators=30)
## r_forest = RandomForestClassifier(criterion='entropy', n_estimators=100)
## r_forest_score = cross_val_score(r_forest, normalized_train_data, train_data['Survived'], cv=3, n_jobs=2)
## print 'random forest cv_score=', r_forest_score
## r_forest.fit(normalized_train_data, train_data['Survived'])
## r_forest_predict = r_forest.predict(normalized_test_data)
## test_data['Survived'] = r_forest_predict
## test_data.to_csv('random_forest_class.csv', columns=['PassengerId',  'Survived'], index=False)

#############  END model training and testing ##############
