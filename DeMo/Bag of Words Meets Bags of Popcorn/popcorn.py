# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:36:59 2018

@author: Administrator
"""

import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer 
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
#unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub('[^a-zA-Z]',' ',review_text)
    words=letters_only.lower().split()
    stops=set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))

def clean_data(df):
    clean_data_reviews = []
    for i in range(0,len(df)):
        if ((i+1)%1000==0):
            print('Review {} of {}'.format(i+1,len(df)))
        clean_data_reviews.append(review_to_words(df['review'][i]))
    return(clean_data_reviews)

df_train = clean_data(train)
df_test = clean_data(test)

def data_vec(df) :
    vectorizer = CountVectorizer(analyzer = 'word',tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)
    data_features = vectorizer.fit_transform(df)
#print(train_data_features.dtype)
    data_features = data_features.toarray() 
#print(train_data_features.dtype) 
#print(train_data_features.shape)
    vocab = vectorizer.get_feature_names()
    return(data_features,vocab)

data_features_train,vocab_train = data_vec(df_train)
data_features_test,vocab_test = data_vec(df_test)

#dist=np.sum(train_data_features,axis=0)
#for tag,count in zip(vocab,dist):
#    print(count,tag)
   

forest = RandomForestClassifier( n_estimators = 100 )
 
print("Fitting a random forest to labeled training data...")
forest = forest.fit( data_features_train, train["sentiment"] )
result = forest.predict( data_features_test )


output = pd.DataFrame({'id':test['id'],'sentiment':result})
output.to_csv( "result.csv", index=False, quoting=3 )
# Test & extract results 
