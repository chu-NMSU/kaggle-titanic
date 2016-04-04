import pandas
import numpy
import seaborn
seaborn.set(style="whitegrid")
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

#############  preprocess ##############
def preprocess_embarked(numeric=True):
    ## convert Embarked, Sex to numeric data
    ## C = Cherbourg; Q = Queenstown; S = Southampton
    train_data['Embarked'] = train_data['Embarked'].fillna('S') # fill missing Embarked
    test_data['Embarked'] = test_data['Embarked'].fillna('S') # fill missing Embarked
    if numeric:
        train_data.loc[train_data['Embarked']=='C', 'Embarked']=0.0
        train_data.loc[train_data['Embarked']=='Q', 'Embarked']=1.0
        train_data.loc[train_data['Embarked']=='S', 'Embarked']=2.0
        test_data.loc[test_data['Embarked']=='C', 'Embarked']=0.0
        test_data.loc[test_data['Embarked']=='Q', 'Embarked']=1.0
        test_data.loc[test_data['Embarked']=='S', 'Embarked']=2.0

# train_data = pandas.read_csv('train_numeric.csv')
train_data = pandas.read_csv('train.csv')
test_data = pandas.read_csv('test.csv')

## fill missing age with median age. TODO explore this later
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Age'] = train_data['Age'].astype(int)
test_data['Age'] = test_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].astype(int)
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].median())

## preprocess Sex
train_data.loc[train_data['Sex']=='male', 'Sex']=0.0
train_data.loc[train_data['Sex']=='female', 'Sex']=1.0
test_data.loc[test_data['Sex']=='male', 'Sex']=0.0
test_data.loc[test_data['Sex']=='female', 'Sex']=1.0
preprocess_embarked(False)

train_embarked_col = train_data['Embarked']
test_embarked_col = test_data['Embarked']
train_data = pandas.get_dummies(train_data, columns=['Embarked'])
test_data = pandas.get_dummies(test_data, columns=['Embarked'])
train_data.insert(train_data.shape[1], 'Embarked', train_embarked_col)
test_data.insert(test_data.shape[1], 'Embarked', test_embarked_col)

# create Has_Family column
train_data.insert(train_data.shape[1], 'Has_Family', 0)
train_data.loc[train_data['SibSp']+train_data['Parch']>0, 'Has_Family'] = 1
test_data.insert(test_data.shape[1], 'Has_Family', 0)
test_data.loc[test_data['SibSp']+test_data['Parch']>0, 'Has_Family'] = 1
#############  END preprocess ##############

train_data.to_csv('train_preprocessed.csv')
test_data.to_csv('test_preprocessed.csv')

labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S','Has_Family'] #
## TODO normalize train test data
normalized_data = pandas.concat([train_data, test_data])[labels]
train_num, test_num = train_data.shape[0], test_data.shape[0]
normalized_data = normalized_data.apply(lambda x: (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))
normalized_train_data = normalized_data[0:train_num]
normalized_test_data = normalized_data[train_num::]
normalized_train_data.insert(0, 'Survived', 0)
normalized_train_data['Survived'] = train_data['Survived']

normalized_train_data.to_csv('norm_train_preprocessed.csv')
normalized_test_data.to_csv('norm_test_preprocessed.csv')

## Basic feature factor plot
## seaborn.factorplot(x='Embarked', y='Survived', hue='Sex', kind="bar", data=train_data)
## seaborn.plt.savefig('survived-embark-sex.pdf')
## seaborn.plt.clf()
## seaborn.countplot(x='Embarked', hue='Sex', data=train_data)
## seaborn.plt.savefig('survived-count-embark-sex.pdf')
## seaborn.plt.clf()

## seaborn.factorplot(x='Pclass', y='Survived', hue='Sex', kind="bar", data=train_data)
## seaborn.plt.savefig('survived-pclass-sex.pdf')
## seaborn.plt.clf()
## seaborn.countplot(x='Pclass', hue='Sex', data=train_data)
## seaborn.plt.savefig('survived-count-pclass-sex.pdf')
## seaborn.plt.clf()
## 
## seaborn.factorplot(x='Has_Family', y='Survived', hue='Sex', kind="bar", data=train_data)
## seaborn.plt.savefig('survived-has-family-sex.pdf')
## seaborn.plt.clf()
## seaborn.countplot(x='Has_Family', hue='Sex', data=train_data)
## seaborn.plt.savefig('survived-count-has-family-sex.pdf')
## seaborn.plt.clf()
