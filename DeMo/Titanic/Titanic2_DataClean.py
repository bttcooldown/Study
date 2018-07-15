# -*- coding: utf-8 -*-
"""
Created on Thu May 24 21:05:06 2018

@author: Administrator
"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import re
import numpy as np
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


data1 = pd.read_csv('train.csv')
data2 = pd.read_csv('test.csv')
data2['Survived'] = 0
data = data1.append(data2);
PassengerId = data2['PassengerId']
#data.drop(['PassengerId','Ticket'],axis=1,inplace=True)

#data.iloc[data['Embarked'].isnull(),'Embarked']='S'
data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
data['Embarked'] = pd.factorize(data['Embarked'])[0]
emb_dum = pd.get_dummies(data['Embarked'],prefix=data[['Embarked']].columns[0])
data = pd.concat([data,emb_dum],axis=1)

data['Sex'] = pd.factorize(data['Sex'])[0]
sex_dum = pd.get_dummies(data['Sex'],prefix=data[['Sex']].columns[0])
data = pd.concat([data,sex_dum],axis=1)

data['Title'] = data['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
data['Title'] = data['Title'].map(title_Dict)
data['Title'] = pd.factorize(data['Title'])[0]
title_dum = pd.get_dummies(data['Title'], prefix=data[['Title']].columns[0])
data = pd.concat([data, title_dum], axis=1)
data['Name_length'] = data['Name'].apply(len)

data['Fare'] = data[['Fare']].fillna(data.groupby('Pclass').transform(np.mean))
#通过对Ticket数据的分析，我们可以看到部分票号数据有重复，同时结合亲属人数及名字的数据，
#和票价船舱等级对比，我们可以知道购买的票中有家庭票和团体票，
#所以我们需要将团体票的票价分配到每个人的头上。
data['Group_Ticket'] = data['Fare'].groupby(data['Ticket']).transform('count')
data['Fare'] = data['Fare'] / data['Group_Ticket']
data.drop(['Group_Ticket'], axis=1, inplace=True)
#使用binning给票价分等级：
data['Fare_bin'] = pd.qcut(data['Fare'],5)
#对于5个等级的票价我们也可以继续使用dummy为票价等级分列：
data['Fare_bin_id'] = pd.factorize(data['Fare_bin'])[0]
fare_dum = pd.get_dummies(data['Fare_bin_id'],prefix=data[['Fare']].columns[0])
data = pd.concat([data,fare_dum],axis=1)
data.drop(['Fare_bin'], axis=1, inplace=True)   

pclass_dum = pd.get_dummies(data['Pclass'],prefix=data[['Pclass']].columns[0])
data = pd.concat([data,pclass_dum],axis=1)
data['Pclass'] = pd.factorize(data['Pclass'])[0]

def family_size_category(family_size):
    if family_size<=1:
        return 'Single'
    elif family_size<=4:
        return 'Small_Family'
    else:
        return 'Large_Family' 

data['family_size'] = data['Parch']+data['SibSp']+1
data['family_size_category'] = data['family_size'].map(lambda x:family_size_category(x))
le_family = LabelEncoder()
le_family.fit(np.array(['Single','Small_Family','Large_Family']))
data['family_size_category'] = le_family.transform(data['family_size_category'])
family_dum = pd.get_dummies(data['family_size_category'],prefix=data[['family_size_category']].columns[0])
data = pd.concat([data,family_dum],axis=1)

miss_age = pd.DataFrame(data[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'family_size', 'family_size_category','Fare', 'Fare_bin_id', 'Pclass']])
miss_age_train = miss_age[miss_age['Age'].notnull()]
miss_age_test = miss_age[miss_age['Age'].isnull()]
#建立Age的预测模型，我们可以多模型预测，然后再做模型的融合，提高预测的精度。
miss_age_X_train = miss_age_train.iloc[:,1:]
miss_age_y_train = miss_age_train.loc[:,['Age']]
miss_age_X_test = miss_age_test.iloc[:,1:]
# model 1  gbm
gbm_reg = GradientBoostingRegressor(random_state=42)
gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
gbm_reg_grid = model_selection.GridSearchCV(gbm_reg,gbm_reg_param_grid)
gbm_reg_grid.fit(miss_age_X_train,miss_age_y_train)
#print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
#print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
#print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
miss_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(miss_age_X_test)
# model 2 rf
rf_reg = RandomForestRegressor()
rf_reg_param_grid = {'n_estimators':[200],'max_depth':[5],'random_state':[0]}
rf_reg_grid = model_selection.GridSearchCV(rf_reg,rf_reg_param_grid)
rf_reg_grid.fit(miss_age_X_train,miss_age_y_train)
miss_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(miss_age_X_test)
# two models merge
miss_age_test.loc[:,['Age']] = np.mean([miss_age_test['Age_GB'], miss_age_test['Age_RF']])
miss_age_test.drop(['Age_GB', 'Age_RF'],axis=1,inplace=True)
data.loc[(data.Age.isnull()),'Age'] = miss_age_test

data['Ticket_Letter'] = data['Ticket'].str.split().str[0]
data['Ticket_Letter'] = data['Ticket_Letter'].apply(lambda x:'U0' if x.isnumeric() else x)
data['Ticket_Letter'] = pd.factorize(data['Ticket_Letter'])[0]

data.loc[(data.Cabin.isnull()),'Cabin'] = 'U0'
data['Cabin'] = data['Cabin'].apply(lambda x: 0 if x=='U0' else 1)

#特征间相关性分析
#Correlation = pd.DataFrame(data[['Embarked', 'Sex', 'Title', 'Name_length', 'family_size', 'family_size_category','Fare', 'Fare_bin_id', 'Pclass', 
#      'Age', 'Ticket_Letter', 'Cabin']])
#colormap = plt.cm.viridis
#plt.figure(figsize=(14,12))
#plt.title('Pearson Correlation of Features', y=1.05, size=15)
#sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#g = sns.pairplot(data[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
#       u'family_size', u'Title', u'Ticket_Letter']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
#g.set(xticklabels=[])

#输入模型前的一些处理：
data[['Age','Fare','Name_length']] = StandardScaler().fit_transform(data[['Age','Fare','Name_length']])        
data_b = data
data.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id',  
                          'Parch', 'SibSp', 'family_size_category', 'Ticket'],axis=1,inplace=True)
data_train = data[:891]
data_test = data[891:]
data_train_X = data_train.drop(['Survived'],axis=1)
data_train_y = data_train['Survived']
data_test_X = data_test.drop(['Survived'],axis=1)



  



        



