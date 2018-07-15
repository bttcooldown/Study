# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:33:11 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4

train = pd.read_csv('D:\Python files\DeMo\GBM\Dataset\Train_nyOWmfK.csv',header=0)
train.drop(train.index[62689],inplace=True)
test = pd.read_csv('D:\Python files\DeMo\GBM\Dataset\Test_bCtAN1w.csv',header=0)
train['source']= 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)
#data.apply(lambda x:sum(x.isnull()))

#Look at categories of all object variables:

#var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source']
#for v in var:
#    print('\nFrequency of feature {}'.format(v))
#    print(data[v].value_counts())


#len(data['City'].unique())
#drop city because too many unique
data.drop('City',axis=1,inplace=True)
data['Age'] = data['DOB'].apply(lambda x:118-int(x[-2:]))
data.drop('DOB',axis=1,inplace=True)
data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)
data.drop('Employer_Name',axis=1,inplace=True)
#data.boxplot(column='Existing_EMI',return_type='axes')
#data['Existing_EMI'].describe()
data['Existing_EMI'].fillna(0, inplace=True)
data['Interest_Rate_Missing'] = data['Interest_Rate'].apply(lambda x:1 if pd.isnull(x) else 0)
data.drop('Interest_Rate',axis=1,inplace=True)
data.drop('Lead_Creation_Date',axis=1,inplace=True)
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)
data['Loan_Amount_Submitted_Missing'] = data['Loan_Amount_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data['Loan_Tenure_Submitted_Missing'] = data['Loan_Tenure_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data.drop(['Loan_Amount_Submitted','Loan_Tenure_Submitted'],axis=1,inplace=True)
data.drop('LoggedIn',axis=1,inplace=True)
data.drop('Salary_Account',axis=1,inplace=True)
data['Processing_Fee_Missing'] = data['Processing_Fee'].apply(lambda x: 1 if pd.isnull(x) else 0)
data.drop('Processing_Fee',axis=1,inplace=True)
data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
data['Var1'].fillna('HBXX',inplace=True)

le = LabelEncoder()
var_to_encode = ['Device_Type','Filled_Form','Gender','Var2','Var1','Mobile_Verified','Source']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
data = pd.get_dummies(data, columns=var_to_encode)

train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']
train.drop('source',axis=1,inplace=True)
test.drop(['source','Disbursed'],axis=1,inplace=True)
train.to_csv(r'D:\Python files\DeMo\GBM\Dataset\train_modified.csv',index=False)
test.to_csv(r'D:\Python files\DeMo\GBM\Dataset\test_modified.csv',index=False)


#def modelfit(alg,dtrain,predictors,performCV=True,printFeatureImportance=True, cv_folds=5):
    







