# Roll No : 18ME10050
# Name : Sandipan Haldar
# Assignment No : 1
# Pandas, Numpy  lib have been used
#pandas have only been used to read the csv file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df1=pd.read_csv('test.csv')
dataset=df.iloc[:,:].reset_index(drop=True)

# Global array to store the training error for each degree
train_errors_list = []

def normalize(df):
    final = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        final[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return final
# Compute the predicted values for given weights and input features
def hypothesis(weights,input_x):
	# Weights = (n+1)x1, n - degree of polynomial
	# input_x = m x (n+1), m - no. of points
	# Output = m x 1 - predicted values for each example
	result = np.matmul(input_x,weights)
	return result

# Mean squared error - cost function
def mean_squared_error(predictions,labels):             
	squared_error = (predictions-labels)**2
	mse = squared_error.mean()
	return mse/2

# Gradient Descent using MSE
def gradient_descent(train_x,train_y,lamda,alpha=0.05,iterations=None):
    num_points = int(train_x.shape[0])
    num_features = int(train_x.shape[1])
    #print(num_points)
    train_y = train_y.reshape(num_points,1)
    weights = np.random.rand(num_features,1) - 0.5
    predictions = hypothesis(weights,train_x)
    if iterations != None:
        for it in range(iterations):
            #updating by only adding the lamda term
            for i in range(num_features):
                weights[i] =weights[i]-(alpha*lamda)/(4*num_points)- (alpha/num_points)*np.sum(np.multiply((predictions-train_y).reshape(num_points,1),train_x[:,i].reshape(num_points,1)))
            predictions = hypothesis(weights,train_x)
            error = mean_squared_error(predictions,train_y)+(lamda/8)*(weights).mean()
   # If no iterations are specified, run for upto convergence with difference 10e-7
    else:
       it=0
       prev_error = 0
       while True:
           it+=1
           for i in range(num_features):
               #update weights acc to gradient descent adding only the lamda term
                weights[i] =weights[i]-(alpha*lamda)/(4*num_points)- (alpha/num_points)*np.sum(np.multiply((predictions-train_y).reshape(num_points,1),train_x[:,i].reshape(num_points,1)))
           predictions=hypothesis(weights,train_x)
           error= mean_squared_error(predictions,train_y)+(lamda/8)*(weights).mean()
           if(prev_error>error and prev_error-error<10e-7):
               break
           prev_error=error
    print('Training error after '+str(it)+' iterations = '+str(error))
    train_errors_list.append(error)
       
    return weights
   
# Given individual values of x and the degree, compute the array [1,x,x^2,..] for each value
def generate_data(data,degree):
    num_points = data.shape[0]
    new_data = np.ones(num_points).reshape(num_points,1)
    for i in range(degree):
        if i==0:
            last_row= np.ones(num_points).reshape(num_points,1)
            new_row = np.multiply(last_row,data).reshape(num_points,1)
            last_row=new_row
            new_data = np.concatenate((new_data,new_row),axis=1)
        else:
            new_row = np.multiply(last_row,data).reshape(num_points,1)
            last_row=new_row
            new_data = np.concatenate((new_data,new_row),axis=1)
    return new_data

def main():
    
    test_error_list=[]
    global train_errors_list
    train_errors_list.clear()
    
    final = normalize(df)
    finaltest = normalize(df1)
    X_train = final.iloc[:,:-1].reset_index(drop=True)
    Y_train =final.iloc[:,-1].reset_index(drop=True)
    X_test = finaltest.iloc[:,:-1].reset_index(drop=True)
    Y_test =finaltest.iloc[:,-1].reset_index(drop=True)
   
    
    #print(X_test)
    X_train = X_train.to_numpy()
    Y_train = Y_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy()
    #print(X_test)
    Y_train=np.array(Y_train)
    Y_test=np.array(Y_test)
    train_y=Y_train
    test_y=Y_test
    trained_weights=[]
    
   # print(X_train.shape[0])
    for lamda in range(1,5): 
        degree=1
        print('+++lamda :' +str(lamda/4))
        for i in range(1,3):
           
            print('degree = '+str(degree))
            train_x = generate_data(X_train,degree)
            #print(train_x)
            test_x = generate_data(X_test,degree)
            
            weights = gradient_descent(train_x,train_y,lamda,iterations=10000)
            trained_weights.append(weights)
            predictions = hypothesis(weights,test_x)
            error = mean_squared_error(predictions,test_y)+(lamda/8)*(weights).mean()
            print('Mean squared error on test data = '+str(error))
            test_error_list.append(error)
            degree+=7
            
    train_errors = np.array(train_errors_list)
    test_errors = np.array(test_error_list)
    train_errors_deg1=[train_errors[0],train_errors[2],train_errors[4],train_errors[6]]
    train_errors_deg8=[train_errors[1],train_errors[3],train_errors[5],train_errors[7]]
    test_errors_deg1=[test_errors[0],test_errors[2],test_errors[4],test_errors[6]]
    test_errors_deg8=[test_errors[1],test_errors[3],test_errors[5],test_errors[7]]
    #print(train_errors)
    
    plt.title('Training error plot after lasso regularization on the deg1 curve')
    plt.xlabel('(lamda*4)')
    plt.ylabel('Training Error')
    lamda = list(range(1,5))
    plt.plot(lamda,train_errors_deg1)
    plt.show()  
    
    plt.title('Training error plot after lasso regularization on the deg8 curve')
    plt.xlabel('(lamda*4)')
    plt.ylabel('Training Error')
    lamda = list(range(1,5))
    plt.plot(lamda,train_errors_deg8)
    plt.show() 
    
    plt.title('test error plot after lasso regularization on the deg1 curve')
    plt.xlabel('(lamda*4)')
    plt.ylabel('TTest Error')
    lamda = list(range(1,5))
    plt.plot(lamda,test_errors_deg1)
    plt.show()  
    
    plt.title('Test error plot after lasso regularization on the deg8 curve')
    plt.xlabel('(lamda*4)')
    plt.ylabel('Test Error')
    lamda = list(range(1,5))
    plt.plot(lamda,test_errors_deg8)
    plt.show()  
    
    plt.title('Test error ans training error plot after lasso regularization on the deg8 curve')
    plt.xlabel('(lamda*4)')
    plt.ylabel('error')
    lamda = list(range(1,5))
    deg8ts=plt.plot(lamda,test_errors_deg8)
    deg8tr=plt.plot(lamda,train_errors_deg8)
    plt.legend([deg8tr[0],deg8ts[0]],('Train Error','Test Error'),loc=0)
    plt.show()  
    
    plt.title('Test error ans training error plot after lasso regularization on the deg1 curve')
    plt.xlabel('(lamda*4)')
    plt.ylabel('error')
    lamda = list(range(1,5))
    deg1ts=plt.plot(lamda,test_errors_deg1)
    deg1tr=plt.plot(lamda,train_errors_deg1)
    plt.legend([deg1tr[0],deg1ts[0]],('Train Error','Test Error'),loc=0)
    plt.show()  
    
        
    
           

if __name__ == '__main__':
	main()    
