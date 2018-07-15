# -*- coding: utf-8 -*-
"""
Created on Tue May 22 21:12:55 2018

@author: Administrator
"""

from sklearn.datasets import load_boston
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.cross_validation import *
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import ensemble

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size = 0.25, random_state=33)

# 正规化的目的在于避免原始特征值差异过大，导致训练得到的参数权重不一
X_train = scale(X_train)
X_test = scale(X_test)

y_train = scale(y_train)
y_test = scale(y_test)

def train_and_evaluate(clf, X_train, y_train):
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ('Average coefficient of determination using 5-fold cross validation:', np.mean(scores))

    
#先用线性模型尝试， SGD_Regressor
clf_sgd= linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)
train_and_evaluate(clf_sgd, X_train, y_train)

#用LinearRegression
lr = linear_model.LinearRegression()
train_and_evaluate(clf_sgd, X_train, y_train)


# 升高维度，效果明显，但是此招慎用@@，特征高的话, CPU还是受不了，内存倒是小事。其实到了现在，连我们自己都没办法直接解释这些特征的具体含义了。
clf_svr_poly = SVR(kernel='rbf')#'poly','linear'
train_and_evaluate(clf_svr_poly, X_train, y_train)

clf_et = ensemble.ExtraTreesRegressor()
train_and_evaluate(clf_et, X_train, y_train)