#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def weight_initialiser(dim1,dim2,dim3,out):
    weight0 = np.random.normal(-1,1,size = (dim1,dim2))
    weight1 = np.random.normal(-1,1,size = (dim2,dim3))
    weight2 = np.random.normal(-1,1,size = (dim3,out))
    return weight0,weight1,weight2

def preprocess():
    df = pd.read_csv('seeds_dataset.txt', sep="\t", header=None)
    
    df = pd.concat([df,pd.get_dummies(df[7], prefix='Att')],axis=1)
    from scipy.stats import zscore
    df=df.apply(zscore)
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
    y_train=dftrain.loc[:,9:]
    X_test=dftest.loc[:,1:7]
    y_train.reset_index()
    y_test=dftest.loc[:,9:]
    X_train=X_train.to_numpy()
    X_test=X_test.to_numpy()
    y_train=y_train.to_numpy()
    y_test=y_test.to_numpy()
    y_train = y_train.reshape(y_train.shape[0],3)
    y_test = y_test.reshape(y_test.shape[0],3)
    y_test = y_test.astype(np.float)
    y_train = y_train.astype(np.float)
    return X_train,X_test,y_train,y_test
def max2(a,b):
    if(a>b):
        return 1
    else: 
        return 0
    
def relu_derivative(x):
    return np.vectorize(max2)(x,0)
def sigmoid_derivative(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def relu(x):
    return np.vectorize(max)(x,0)
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def softmax(x):
    num = np.exp(x)
    den = np.sum(np.exp(x),axis=1)
    den = den.reshape(den.shape[0],1)
    return num/den

    

def forward_pass(X_train,weight0,weight1,weight2):
    layer0=X_train
    layer1=relu(np.dot(layer0,weight0))
    layer2 = sigmoid(np.dot(layer1,weight1))
    layer3=softmax(np.dot(layer2,weight2))
    return layer0,layer1,layer2,layer3

def backprop(X_train,y_train,layer0,layer1,layer2,layer3,weight0,weight1,weight2):
    d3 = y_train - layer3
    dweights3 = np.dot(layer2.T,d3)
    d2 = np.dot(d3,weight2.T)*relu_derivative(np.dot(layer1,weight1))
    dweights2 = np.dot(layer1.T,d2)
    d1 = np.dot(d2,weight1.T)*relu_derivative(np.dot(X_train,weight0))
    dweights1 = np.dot(X_train.T,d1)
    return dweights1,dweights2,dweights3

def cross_entropy(pred,label):
    epsilon = 1e-6
    temp = label[:,:1]*np.log(pred[:,:1]+epsilon)+(label[:,1:])*(np.log((pred[:,1:])+epsilon))
    return (-1 * np.sum(temp))/len(label)
def predict(X_test,y_test,weight0,weight1,weight2):
    layer_0=X_test
    layer_1=relu(np.dot(layer_0,weight0))
    layer_2=relu(np.dot(layer_1,weight1))
    layer_3 = softmax(np.dot(layer_2,weight2))
    correct=0
    maxi=np.amax(layer_3,axis=1)
    y_test_new=y_test
    for i in range(len(y_test)):
       for j in range(3):
           if(y_test_new[i][j]>0):
               y_test_new[i][j]=1
           else:
               y_test_new[i][j]=0
    for i in range(len(layer_3)):
       for j in range(3):
           if(layer_3[i][j]==maxi[i]):
               layer_3[i][j]=1
           else:
               layer_3[i][j]=0
       if(np.array_equal(layer_3[i],y_test_new[i])):
            correct += 1
    return correct/len(layer_3)
    
def train(X_train,y_train,weight0, weight1,weight2,X_test,y_test, alpha = 0.01,epsilon = 0.05,num_batches=4):
    it=0
    train_errors=[]
    test_errors=[]
    train_acc=[]
    test_acc=[]
    while True:
        it+=1
        l= [int(i*len(X_train)/num_batches) for i in range(num_batches+1)]
        pred=[]
        
        for i in range(num_batches):
            layer0,layer1,layer2,layer3=forward_pass(X_train[l[i]:l[i+1]],weight0,weight1,weight2)
            pred.extend(layer3)
            delta1,delta2,delta3=backprop(X_train[l[i]:l[i+1]],y_train[l[i]:l[i+1]],layer0,layer1,layer2,layer3,weight0,weight1,weight2)
            weight0+=(alpha/len(X_train))*delta1
            weight1+=(alpha/len(X_train))*delta2
            weight2+=(alpha/len(X_train))*delta3
        error=cross_entropy(np.array(pred),y_train)
        tr=predict(X_train,y_train,weight0,weight1,weight2)
        ltest0,ltest1,ltest2,ltest3=forward_pass(X_test,weight0,weight1,weight2)
        test_error=cross_entropy(ltest3,y_test)
        te=predict(X_test,y_test,weight0,weight1,weight2)
        print('iteration = ',it,' trainerror  = ',error,' testerror = ',test_error)
        train_errors.append(error)
        train_acc.append(tr)
        test_acc.append(te)
        test_errors.append(test_error)
        if((it>200)):
            break
    ephocs = [i+1 for i in range(len(train_errors))]
    plt.title('Train and test errors wrt epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Cross entropy error')
    tr = plt.plot(ephocs,train_errors)
    te = plt.plot(ephocs,test_errors)
    plt.legend([tr[0],te[0]],('Train Error','Test Error'),loc=0)
    plt.show()
    plt.title('Train and test accuracy wrt epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    tr = plt.plot(ephocs,train_acc)
    te = plt.plot(ephocs,test_acc)
    plt.legend([tr[0],te[0]],('Train acc','Test acc'),loc=0)
    plt.show()
    return weight0,weight1,weight2
            
    
        
            
        
    
    
    
def main():
    preprocess()
    X_train,X_test,y_train,y_test=dataloader()
    dim1 = 7
    dim2=64
    dim3=32
    out=3
    weight0,weight1,weight2=weight_initialiser(dim1,dim2,dim3,out)
    alpha =0.01
    weight0,weight1,weight2=train(X_train,y_train,weight0,weight1,weight2,X_test,y_test,alpha=alpha,num_batches=4)
    tr=predict(X_train,y_train,weight0,weight1,weight2)
    te=predict(X_test,y_test,weight0,weight1,weight2)
    print('Train accuracy(1B) =  '+str(tr * 100.0 ))
    print('Train accuracy(1B) =  '+str(te * 100.0 ))
    
    
    
if __name__=="__main__":
    main()



