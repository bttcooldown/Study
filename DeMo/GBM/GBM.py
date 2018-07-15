# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 23:05:48 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation,metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,4

train = pd.read_csv(r'D:\Python files\DeMo\GBM\Dataset\train_modified.csv')
IDcol ='ID'
target = 'Disbursed'
def modelfit(alg,dtrain,predictors,performCV=True,printFeatureImportance=True,cv_folds=5):
    alg.fit(dtrain[predictors],dtrain[target])
    predictions = alg.predict(dtrain[predictors])
    predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    if performCV:
        cv_score = cross_validation.cross_val_score(alg,dtrain[predictors],dtrain[target],cv=cv_folds,scoring='roc_auc')
    print('\nModel Report')
    print('Accuracy : {}'.format(metrics.accuracy_score(dtrain['Disbursed'].values, predictions)))
    print('AUC Score(Train):{}'.format(metrics.roc_auc_score(dtrain['Disbursed'],predprob)))
    
    if performCV:
        print("CV Score : Mean - {} | Std - {} | Min - {} | Max - {}".format(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_,predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar',title='Feature Importance')

predictors = [x for x in train.columns if x not in [target,IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0,train,predictors)

#固定learning rate，预设值先决定出树的个数        
predictors = [x for x in train.columns if x not in [target,IDcol]]
param_test1 = {'n_estimators':np.arange(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
                        min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),
                        param_grid = param_test1,scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

#{'n_estimators': 70} 0.8356590800604575    
   
predictors = [x for x in train.columns if x not in [IDcol,target]]
param_test2 = {'max_depth':np.arange(5,16,2),'min_samples_split':np.arange(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,n_estimators=70,
                        max_features='sqrt', subsample=0.8, random_state=10),
                        param_grid = param_test2,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch2.fit(train[predictors],train[target])
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

#{'max_depth': 7, 'min_samples_split': 600} 0.8393818844629726









        