#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import copy
#reading the dataset
df=pd.read_csv("AllBooks_baseline_DTM_Labelled.csv")
df=df.drop([df.index[13]])
split_names = (df['Unnamed: 0']
    .str.strip()
    .str.split('_', n=1, expand=True)
    .rename(columns={0:'text', 1:'chapter'}))
df = pd.concat([ split_names,df], axis=1)
df = df.drop(df.columns[[ 1, 2]], axis=1)

dtm=df.iloc[:,1:].values
tf=np.zeros(8266)
A=dtm[:,:]
#forming the tfidf matrix
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
#normmalising
def mag(vec1):
    tot=0
    for i in range(round(len(vec1))):
        tot+=pow(vec1[i], 2)
    magnitude = tot**0.5
    return magnitude

magnitude=np.zeros(589)
for i in range(589):
    magnitude[i]=mag(tfidf[i])
for i in range(589):
    for j in range(8265):
        tfidf[i][j]=tfidf[i][j]/magnitude[i] 
#applying the PCA        
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
reduced_tfidf = pca.fit_transform(tfidf)


def distance(vec1, vec2):
    simil=0
    for i in range(0,100):
        simil += (vec1[i] * vec2[i])
    d=simil/(mag(vec1)*mag(vec2))
    dist=np.exp(-(d/np.power(8266,2)))
    return dist



#finding the cosine similarity 
"""
cos_sim=np.zeros([589,589])
for i in range(589):
    for j in range(i,589):
        tot=0
        for k in range(100):
            tot+=(tfidf[i][k]*tfidf[j][k])
          
        cos_sim[i][j]=tot/(mag[i]*mag[j])
        
#finding the distance matrix
dist=np.zeros([589,589])
for i in range (589):
    for j in range(i,589):
        dist[i][j]=np.exp(-(cos_sim[i][j]/np.power(100,2)))
"""        
np.savetxt('tfidf.csv',tfidf)
    

def single_linkage(temp_clusters):
    num_clusters = len(temp_clusters)
    while(num_clusters > 8):
        closest_val = 99999
        closest_i = 0
        closest_j = 1
        for i in range(0,num_clusters-1):
            for j in range(i+1,num_clusters):
                curr_min = 99999
                for k in range(len(temp_clusters[i])):
                    for l in range(len(temp_clusters[j])):
                        dist_here = distance(temp_clusters[i][k],temp_clusters[j][l])
                        if(dist_here < curr_min):
                            curr_min = dist_here
                if(curr_min <= closest_val):
                    closest_val = curr_min
                    closest_i = i
                    closest_j = j
        temp_clusters[closest_i].extend(temp_clusters[closest_j])
        temp_clusters.pop(closest_j)
        num_clusters = len(temp_clusters)
    return temp_clusters

dist1=np.loadtxt('tfidf.csv')
dist_list=dist1.tolist()
init_single_clusters = copy.deepcopy(dist_list)
single_clusters=single_linkage(init_single_clusters)
for i in range(len(single_clusters)):
		print("cluster "+str(i)+' size = '+str(len(single_clusters[i])))
		print(*single_clusters[i],sep=',')
        
with open('reduced_agglomerative.txt', 'w') as f:
   for i in range(len(single_clusters)):
        f.write("cluster "+str(i)+' size = '+str(len(single_clusters[i]))+'\n')
        f.write(str(single_clusters[i]))
        f.write('\n\n')
