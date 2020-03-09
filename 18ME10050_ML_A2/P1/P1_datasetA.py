#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

df=pd.read_csv("winequality-red.csv",sep=";")
# Adjusting the column Quality
df.loc[df["quality"] <= 6, "quality"] = 0
df.loc[df["quality"] > 6, "quality"] = 1
#Normalising the dataset
df=((df-df.min())/(df.max()-df.min()))
df.to_csv('dataset_A.csv') 


