#!/usr/bin/env python3
# -*- coding: utf-8 -*


import numpy as np
import pandas as pd

#reading the dataset
df=pd.read_csv("AllBooks_baseline_DTM_Labelled.csv")
df=df.drop([df.index[13]])
split_names = (df['Unnamed: 0']
    .str.strip()
    .str.split('_', n=1, expand=True)
    .rename(columns={0:'text', 1:'chapter'}))
df = pd.concat([ split_names,df], axis=1)
df = df.drop(df.columns[[ 1, 2]], axis=1)
#forming the tfidf matrix
dtm=df.iloc[:,1:].values
tf=np.zeros(8266)
A=dtm[:,:]
tf=np.sum(A,axis=0)
print(tf)
dft=np.zeros(8266)
idf=np.zeros(8266)
for i in range(8266):
    dft[i]=np.count_nonzero(A[:,i:i+1])
    idf[i]=np.log2(590/(1+dft[i]))
tfidf=np.zeros([589,8265])
for i in range(589):
    for j in range(8265):
        tfidf[i][j]=A[i][j]*idf[j]
 
import sklearn.preprocessing
sklearn.preprocessing.normalize(tfidf, norm='l2', axis=1, copy=True, return_norm=False)
#randomly initialising the means of the clusters       
def InitializeMeans(data, n):
    means = []
    ind_list = []
    for i in range(len(data)):
        ind_list.append(i) 
    indices = np.random.choice(ind_list, size = n)
    for i in range(n):
        means.append(data[indices[i]])
    return means

def mag(vec1):
    tot=0
    for i in range(len(vec1)):
        tot+=pow(vec1[i], 2)
    magnitude = tot**0.5
    return magnitude

        
#finding the distance between two vectors according to the given formula        
def distance(vec1, vec2):
    simil=0
    for i in range(len(vec1)):
        simil += (vec1[i] * vec2[i])
    d=simil/(mag(vec1)*mag(vec2))
    dist=np.exp(-(d/np.power(8266,2)))
    return dist
    
# which cluster the data point belongs to
def which_cluster(features, means):
    dist_min = 100000
    for i in range(len(means)):
        dist = distance(features, means[i])
        if dist <= dist_min:
            dist_min = dist
            cluster = i
    return cluster
# get the corresponding clusters of entire data
def get_clusters(data, means):
    cluster_val = []
    for i in range(len(data)):
        val = which_cluster(data[i], means)
        cluster_val.append(val)
    return cluster_val

# get unique elements in a list
def unique(list1): 
    unique_list = []  
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list
        
# to update the new cluster means
def update_means(data, cluster_val):
    values = unique(cluster_val)
    values.sort()
    means = []
    for i in range(len(values)):
        count = 0
        lst = []
        for j in range(len(data)):
            if cluster_val[j] == values[i]:
                if len(lst) == 0:
                    for k in range(len(data[j])):
                        lst.append(data[j][k])
                    count = count + 1
                    continue
                for k in range(len(lst)):
                    lst[k] += data[j][k]
                count = count + 1
        for k in range(len(lst)):
            lst[k] /= count
        if len(lst) == 0:
            continue
        means.append(lst)
    return means
# get lists with indices of data in different clusters
def sort_clusters(values):
    c_val = unique(values)
    c_val.sort()
    clusters = []
    for i in range(len(c_val)):
        lst = []
        for j in range(len(values)):
            if values[j] == c_val[i]:
                lst.append(j)
        clusters.append(lst)
    return clusters

#storing magnitude of all vectors    
magnitude=np.zeros(589)
for i in range(589):
    magnitude[i]=mag(tfidf[i])
 #normalising       
for i in range(589):
    for j in range(8265):
        tfidf[i][j]=tfidf[i][j]/magnitude[i]       
np.savetxt('tfidf.csv',tfidf)
dist1=np.loadtxt('tfidf.csv')
data=dist1.tolist()   
num_clu = 8
num_iter=100
cluster_means = InitializeMeans(data, num_clu)  
#clustering the data 
for i in range(num_iter): 
    cluster_value = get_clusters(data, cluster_means) # get clusters
    cluster_means = update_means(data, cluster_value) # update cluster means
cluster_value = get_clusters(data, cluster_means) 

for i in range(len(cluster_means)):
	print("Cluster ", i + 1, " mean : ", cluster_means[i])
	print("")    
    
formed_clusters = sort_clusters(cluster_value)
for i in range(len(formed_clusters)):
		print("cluster "+str(i)+' size = '+str(len(formed_clusters[i])))
		print(*formed_clusters[i],sep=',')

with open('kmeans.txt', 'w') as f:
   for i in range(len(formed_clusters)):
        f.write("cluster "+str(i)+' size = '+str(len(formed_clusters[i]))+'\n')
        f.write(str(formed_clusters[i]))
        f.write('\n\n')