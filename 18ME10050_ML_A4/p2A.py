#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
def preprocess():
    df = pd.read_csv('seeds_dataset.txt', sep="\t", header=None)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv('train.csv')
    test.to_csv('test.csv')
def dataloader():
    dftrain=pd.read_csv('train.csv', header=None)
    dftest=pd.read_csv('test.csv', header=None)
    dftrain=dftrain.drop(0,axis=1)
    dftest=dftest.drop(0,axis=1)
    dftrain=dftrain.drop(0,axis=0)
    dftest=dftest.drop(0,axis=0)
    X_train=dftrain.loc[:,1:7]
    y_train=dftrain.loc[:,8]
    X_test=dftest.loc[:,1:7]
    y_test=dftest.loc[:,8]
    
    X_train=X_train.to_numpy()
    X_test=X_test.to_numpy()
    y_train=y_train.to_numpy()
    y_test=y_test.to_numpy()
    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    y_test = y_test.astype(np.float)
    y_train = y_train.astype(np.float)
    return X_train,X_test,y_train,y_test
    
def main():
    preprocess()
    X_train,X_test,y_train,y_test=dataloader()
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation='logistic', solver='sgd', alpha=.01,batch_size=32,max_iter=200)
    clf.fit(X_train,y_train)
    #print(clf.n_iter_)
    y_pred=clf.predict(X_train)
    print('Train accuracy(2A) =  '+str(accuracy_score(y_train,y_pred)*100))
    y_pred=clf.predict(X_test)
    print('Test accuracy(2A) =  '+str(accuracy_score(y_test,y_pred)*100))
    
    
    
    
if __name__=="__main__":
    main()


