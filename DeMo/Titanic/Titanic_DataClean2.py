# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 19:29:23 2018

@author: Administrator
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import pandas as pd
import re

def clean_data(df, drop_passenger_id):
    
    #随机森林算法补充缺失的Age数据
    age_df = df[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df[age_df.Age.notnull()]
    age_df_isnull = age_df[age_df.Age.isnull()]
    X = age_df_notnull[:,1:]
    y = age_df_notnull[:,0]
    RFR = RandomForestRegressor(n_estimators=100,n_jobs=-1)
    RFR.fit(X,y)
    Age_pre = RFR.predict(age_df_isnull.values[:,1:])
    df[df.Age.isnull()]['Age'] = Age_pre 
    
    #虚拟变量
    df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
    embark_dummies = pd.get_dummies(df['Embarked'])
    df = df.join(embark_dummies)
    df = df.drop(['Embarked'],axis=1,inspace=True)
    embark_dummies = df[['S', 'C', 'Q']]
    
    df['Sex'] = pd.factorize(df['Sex'])[0]
    sex_dummies_df = pd.get_dummies(df['Sex'],prefix=df[['Sex']].columns[0])
    df = pd.concat([df,sex_dummies_df],axis=1)

    df.Cabin = df.Cabin.fillna('U0')
    df['Cabin_val'] = df['Cabin'].map(lambda x: re.compile('([a-zA-Z])').search(x).group())
    df['Cabin_val'] = pd.factorize(df['Cabin_val'])[0] #Factorizing
    
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))
    
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    title_dict={}
    title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'],'Officer'))
    title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'],'Royalty'))
    title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'],'Mrs'))
    title_dict.update(dict.fromkeys(['Mlle', 'Miss'],'Miss'))
    title_dict.update(dict.fromkeys(['Master','Jonkheer'],'Master'))
    title_dict.update(dict.fromkeys(['Mr'], 'Mr'))
    df['Title'] = df['Title'].map(title_dict)
    df['Title'] = pd.factorize(df['Title'])[0]
    title_dummies_df = pd.get_dummies(df['Title'])
    df = pd.concat([df,title_dummies_df],axis=1)
    df['Name_length'] = df['Name'].apply(len)
    
    df['Fare'] = df[['Fare']].fillna(df.groupby('Pclass').transform(np.mean))
    
    













       