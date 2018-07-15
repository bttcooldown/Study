# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:28:09 2018

@author: Administrator
"""
import re
import pandas as pd
import numpy as np
import pylab as plt
from Titanic_DataClean import clean_data

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
#培训数据功能，跳过第一列“存活"
df_train = pd.read_csv('train.csv')
train_data = clean_data(df_train, drop_passenger_id=False)
train_features = train_data.values[:, 2:]
# 存活列
train_target = train_data.values[:, 1]
# Fit the model to our training data
clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
print("Mean accuracy of Random Forest: {0}".format(score))

#Random Forest: Predicting
df_test = pd.read_csv('test.csv')
df_test = clean_data(df_test, drop_passenger_id=False)
test_data = df_test.values #数据为测试集争论，并将其转换为一个numpy数组。

#获取测试数据特性，跳过第一列“乘客id”
test_x = test_data[:, 1:]
#预测测试数据的生存值
text_y = clf.predict(test_x)

#Evaluate Model Accuracy
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# Split 80-20 train vs test data
train2_x, test2_x, train2_y, test2_y = train_test_split(train_features, 
                                                    train_target, 
                                                    test_size=0.20, 
                                                    random_state=0)

clf.fit(train2_x,train2_y)
test2_pre = clf.predict(test2_x)
print('Accuracy=%.2f' % ((accuracy_score(test2_y,test2_pre)))) 


