#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

#calculating the magnitude of each vector
mag =np.zeros(589)
for i in range(589):
    tot=0
    for j in range(8265):
        tot=tot+(tfidf[i][j]*tfidf[i][j])
    mag[i]=tot**0.5
#finding the cosine similarity    
cos_sim=np.zeros([589,589])
for i in range(589):
    for j in range(i,589):
        tot=0
        for k in range(8265):
            tot+=(tfidf[i][k]*tfidf[j][k])
            
        cos_sim[i][j]=tot/(mag[i]*mag[j])
        
#finding the distance matrix    
dist=np.zeros([589,589])
for i in range (589):
    for j in range(i,589):
        dist[i][j]=np.exp(-(cos_sim[i][j]/np.power(8266,2)))
np.savetxt('dist.csv',dist)
    

def single_linkage(temp_clusters,num):
    num_clusters = len(temp_clusters)
    data_pts=[]
    for i in range (num_clusters):
        temp=[i]
        data_pts.append(temp)
        
    while(num_clusters > num):
        closest_val = 0
        closest_i = 0
        closest_j = 1
        for i in range(0,num_clusters-1):
            for j in range(i+1,num_clusters):
                curr_min = temp_clusters[i][j]
                
                if(curr_min <= closest_val):
                    closest_val = curr_min
                    closest_i = i
                    closest_j = j
        #temp_clusters[closest_i].extend(temp_clusters[closest_j])
        data_pts[closest_i].extend(data_pts[closest_j])
        data_pts.pop(closest_j)
        for k in range(num_clusters-1):
            temp_clusters[closest_i][k]=min(temp_clusters[closest_i][k],temp_clusters[closest_j][k])
            
        temp_clusters.pop(closest_j)
        num_clusters = len(temp_clusters)
        if(num_clusters==num):
            break         
    return data_pts


dist1=np.loadtxt('dist.csv')
dist_list=dist1.tolist()
single_clusters=single_linkage(dist_list,8)
for i in range(len(single_clusters)):
		print("cluster "+str(i)+' size = '+str(len(single_clusters[i])))
		print(*single_clusters[i],sep=',')
        
with open('‘agglomerative.txt', 'w') as f:
   for i in range(len(single_clusters)):
        f.write("cluster "+str(i)+' size = '+str(len(single_clusters[i]))+'\n')
        f.write(str(single_clusters[i]))
        f.write('\n\n')
