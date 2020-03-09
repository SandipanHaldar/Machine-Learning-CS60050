#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import zscore
#reading the csv file
df=pd.read_csv("winequality-red.csv",sep=";")
# Adjusting the column Quality
df.loc[df["quality"] < 5, "quality"] = 0
df.loc[df["quality"] > 6, "quality"] = 2
df.loc[df["quality"] > 2, "quality"] = 1
# normalising

df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']] = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']].apply(zscore)

#segregating the columns into bins
df['fixed acidity'] = pd.qcut(df['fixed acidity'], 4, labels=[0,1,2,3])
df['volatile acidity'] = pd.qcut(df['volatile acidity'], 4, labels=[0,1,2,3])
df['citric acid'] = pd.qcut(df['citric acid'], 4, labels=[0,1,2,3])
df['residual sugar'] = pd.qcut(df['residual sugar'], 4, labels=[0,1,2,3])
df['chlorides'] = pd.qcut(df['chlorides'], 4, labels=[0,1,2,3])
df['free sulfur dioxide'] = pd.qcut(df['free sulfur dioxide'], 4, labels=[0,1,2,3])
df['total sulfur dioxide'] = pd.qcut(df['total sulfur dioxide'], 4, labels=[0,1,2,3])
df['density'] = pd.qcut(df['density'], 4, labels=[0,1,2,3])
df['pH'] = pd.qcut(df['pH'], 4, labels=[0,1,2,3])
df['sulphates'] = pd.qcut(df['sulphates'], 4, labels=[0,1,2,3])
df['alcohol'] = pd.qcut(df['alcohol'], 4, labels=[0,1,2,3])

df.to_csv('dataset_B.csv') 
