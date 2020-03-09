
#importing the libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#reading the dataset
df=pd.read_csv("dataset_B.csv")
df=df.drop(columns=['Unnamed: 0'])

#forming the dependent and independent variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#Building the classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',min_samples_split=10)
classifier.fit(X, y)

#forming the numpyarray from the dataframe
X=X.iloc[:,:].values
y=df.iloc[:, -1].values

#performing three fold cross validation on randomly formed training and test set
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('+++')
    print('Mean Accuracy(scikit)= '+str(accuracy_score(y_test, y_pred)))
    print('Precision(scikit)= '+str(precision_score(y_test, y_pred, average="macro")) )
    print('Recall(scikit) = '+str(recall_score(y_test, y_pred, average="macro"))) 
