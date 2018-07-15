# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:48:30 2018

@author: Administrator
"""

from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_top_features(data_train_X,data_train_y,n_features):
#    random forest
    rf_est = RandomForestClassifier()
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est,rf_param_grid)
    rf_grid.fit(data_train_X,data_train_y)
    feature_imp_sorted_rf = pd.DataFrame({'feature':list(data_train_X),
                                          'importance':rf_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(n_features)['feature']
    
    # AdaBoost
    ada_est = AdaBoostClassifier()
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est,ada_param_grid)
    ada_grid.fit(data_train_X,data_train_y)
    feature_imp_sorted_ada = pd.DataFrame({'feature':list(data_train_X),
                                           'importance':ada_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(n_features)['feature']
    
    # ExtraTree
    et_est = ExtraTreesClassifier()
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est,et_param_grid)
    et_grid.fit(data_train_X,data_train_y)
    feature_imp_sorted_et = pd.DataFrame({'feature':list(data_train_X),
                                          'importance':et_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(n_features)['feature']
        
    # GradientBoosting
    gb_est =GradientBoostingClassifier()
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid)
    gb_grid.fit(data_train_X,data_train_y)
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(data_train_X),
                                           'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(n_features)['feature']
    
    # DecisionTree
    dt_est = DecisionTreeClassifier()
    dt_param_grid = {'min_samples_split':[2,4],'max_depth':[20]}
    dt_grid = model_selection.GridSearchCV(dt_est,dt_param_grid)
    dt_grid.fit(data_train_X,data_train_y)
    feature_imp_sorted_dt = pd.DataFrame({'feature':list(data_train_X),
                                          'importance':dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    feature_top_n_dt = feature_imp_sorted_dt.head(n_features)['feature']
    
    # merge the n models
    features_top_n = pd.concat([features_top_n_rf,features_top_n_ada,features_top_n_et,features_top_n_gb,feature_top_n_dt],ignore_index=True).drop_duplicates()
    features_importance = pd.concat([feature_imp_sorted_rf,feature_imp_sorted_ada,feature_imp_sorted_et,feature_imp_sorted_gb,feature_imp_sorted_dt],ignore_index=True)
    
    return features_top_n , features_importance

#特征筛选
n_features = 20
feature_top_n, feature_importance = get_top_features(data_train_X,data_train_y,n_features)
data_train_X = data_train_X[feature_top_n]
data_test_X = data_test_X[feature_top_n]

ntrain = data_train_X.shape[0]
ntest = data_test_X.shape[0]
NFOLDS = 7
SEED = 0
kf = KFold(n_splits=NFOLDS,random_state=SEED,shuffle=False)
def get_out_fold(clf,x_train,y_train,x_test):
    oof_train = np.zeros(ntrain)
    oof_test = np.zeros(ntest)
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_index,test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        x_te = x_train[test_index]
        y_tr = y_train[train_index]
        
        clf.fit(x_tr,y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)
        
    oof_test[:]  = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)        



rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)

et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)

dt = DecisionTreeClassifier(max_depth=8)

knn = KNeighborsClassifier(n_neighbors = 2)

svm = SVC(kernel='linear', C=0.025)

x_train = data_train_X.values
y_train = data_train_y.values
x_test = data_test_X.values

rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost 
et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees
gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost
dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree
knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test) # KNeighbors
svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector

x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

from xgboost import XGBClassifier

gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                        colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


with open(r'D:\Python files\DeMo\Titanic\prediction.csv','w') as f:
    for p in predictions:
        f.write(str(p))
        f.write('\n')
    
















    