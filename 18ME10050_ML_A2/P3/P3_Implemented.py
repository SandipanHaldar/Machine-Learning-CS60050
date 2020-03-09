
#importing the libraries
import numpy as np
import pandas as pd
import math

#defining the entropy
def entropy(Y):  
    Ent = 0
    values = Y.unique()
    for val in values:
        frac = Y.value_counts()[val] / len(Y)
        Ent = Ent - frac * math.log(frac)
    return Ent

#claculating the information gain
def gain(X, Y, att):

	Ent_X = entropy(Y)
	values = X[att].unique()
	Ent_sum = 0
	for val in values:
		index = X.index[X[att] == val].tolist()
		Y_temp = Y.iloc[index]
		Y_temp = Y_temp.reset_index(drop=True)
		frac = len(Y_temp)/len(Y)
		Ent_sum = Ent_sum + frac * entropy(Y_temp)
	return (Ent_X - Ent_sum)

#deciding the attribute of current node
def decide_att(X, Y, parent_att):
	attribute = None
	_gain = -100000
	for att in X.keys():
		temp = gain(X, Y, att)
		if temp > _gain:
			if (att in parent_att):
				continue
			_gain = temp
			attribute = att
	if attribute is None:
		return parent_att[-1]
	return attribute

#returns the data of subtree
def get_sub_data(X, Y, att, val):

	index = X.index[X[att] == val].tolist()
	X_temp = X.iloc[index, : ]
	Y_temp = Y.iloc[index]
	X_temp = X_temp.reset_index(drop=True)
	Y_temp = Y_temp.reset_index(drop=True)
	return X_temp, Y_temp

#building the tree using recursion
    
def get_tree(X, Y, parent_att, count, tree = None):
	current_att = decide_att(X,Y,parent_att)
	values = X[current_att].unique()
	if tree is None:                    
		tree = {}
		tree[current_att] = {}
	for val in values:
		X_sub, Y_sub = get_sub_data(X, Y, current_att, val)
		y_values = Y_sub.unique()
		class_count = {}
		for y_val in y_values:
			class_count[y_val] = Y_sub.value_counts()[y_val]
		maximum = max(class_count, key=class_count.get)
		total = 0
		for i in class_count.values():
			total = total + i
		if (count <= 1):
			tree[current_att][val] = maximum
		elif(class_count[maximum]/total == 1):
			tree[current_att][val] = maximum
		else:
			new_parents = parent_att.copy()
			new_parents.append(current_att)
			tree[current_att][val] = get_tree(X_sub, Y_sub, new_parents, count-1)
	return tree
    

#finding the accuracy
def test_accuracy(ptX, ptY, dic, level):
    if type(dic)!=dict:
        if (dic == ptY):
            return 1
        else:
            return 0
    for key in dic:
        value = ptX[key]
        val = dic[key]
        if type(val)==dict:
            if value in val:
                ret_val = test_accuracy(ptX, ptY, val[value], level+1)
                return ret_val
            else:
                avg = []
                for i in val:
                    return avg.append(test_accuracy(ptX, ptY, val[i], level+1))
                if (avg.count(1) >= avg.count(0)):
                    return 1
                else:
                    return 0

#reading the dataset                
df=pd.read_csv("dataset_B.csv")
df=df.drop(columns=['Unnamed: 0'])

#performing the crossvalidation and finding accuracy
for i in range(3):
    shuffled_indices=np.random.permutation(len(df))
    test_set_size= int(len(df)*0.33)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    X_train = df.iloc[train_indices,:-1].reset_index(drop=True)
    X_test = df.iloc[test_indices,:-1].reset_index(drop=True)
    Y_train = df.iloc[train_indices,-1].reset_index(drop=True)
    Y_test = df.iloc[test_indices,-1].reset_index(drop=True)
    parents = []
    tree = get_tree(X_train,Y_train, parents, 11, None)
    test_pts = X_test.to_dict(orient='records')
    print("+++")
    true = 0
    false = 0
    for i in range(len(test_pts)):
        pred = test_accuracy(test_pts[i], Y_test[i], tree, 0)
        if (pred == 1): 
            true = true + 1
        else:
            false = false + 1
        accuracy = (true) / (true + false)
    print("Mean Accuracy (implemented): ", accuracy) 
             