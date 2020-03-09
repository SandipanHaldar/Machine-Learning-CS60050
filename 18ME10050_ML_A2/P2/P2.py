#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#importing the Dataset
df=pd.read_csv("dataset_A.csv")
df=df.drop(columns=['Unnamed: 0'])
#preparing the X and y tables
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
#implimeting logistic regression
def sigmoid(z):
    return 1/(1+ np.exp(-z))
def hypothesis(w,X):
    z = np.array(w[0]+w[1]*X[:,0]+w[2]*X[:,1]+w[3]*X[:,2]+w[4]*X[:,3]+w[5]*X[:,4]+w[6]*X[:,5]+w[7]*X[:,6]+w[8]*X[:,7]+w[9]*X[:,8]+w[10]*X[:,9]+w[11]*X[:,10])
    return z
def cost(w,X,y):
    y_val= hypothesis(w,X)
    y_val=sigmoid(y_val)
    #print(y_val)
    return (-1*(sum(y*np.log(y_val)+(1-y)*np.log(1-y_val))))/X.shape[0]
def grad(w,X,y):
    y_val=hypothesis(w,X)
    y_val=sigmoid(y_val)
    g=np.random.rand(12,1)*0
    g[0]=sum(y_val-y)
    for i in range(1,12):
        g[i]=sum(y_val*X[:,i-1]- y*X[:,i-1])
    return g
def descent(w_new,w_prev,alpha,m):
    j=0
    while True:
        w_prev=w_new
        temp=[0]*12
        for i in range(12):
           temp[i]=w_prev[i]- (alpha/m)*grad(w_prev,X,y)[i]
        
        w_new=temp
        tot =0
        for i in range(12):
            tot+=(w_new[i]-w_prev[i])**2
        if tot < 10e-7:
           
            return w_new
        
        if j>9999:
            
            return w_new
        j+=1
        


#implimenting the scikit learn model
#saga solver
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression( solver='saga',C=1e42)
classifier.fit(X, y) 

"""here penalty ='none' is not being accepted so I set C to a large value
   C is inversely proportional to the lambda of regulazation so if C is large 
   regularization is negligible
   
"""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    w = np.random.rand(12,1)-.5
    m=X_train.shape[0]
    w= descent(w,w,.001,m)
    z=hypothesis(w,X_test)
    s=sigmoid(z)
    threshold = 0.5
    s=np.where(s>threshold, 1, 0)
    print('+++')
    print('Mean Accuracy(scikit)= '+str(accuracy_score(y_test, y_pred))+ '  Mean Accuracy(implemented)= ' +str(accuracy_score(y_test, s)) )
    print('Precision(scikit)= '+str(precision_score(y_test, y_pred, average="macro"))+ '  Precision(implemented)= '+str(precision_score(y_test,s, average="macro")) )
    print('Recall(scikit) = '+str(recall_score(y_test, y_pred, average="macro"))+ '  Recall(implemented) = '+str(recall_score(y_test, s, average="macro"))) 