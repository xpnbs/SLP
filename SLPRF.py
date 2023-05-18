# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:45:13 2018

@author: 13327
"""
from openpyxl import load_workbook
from openpyxl import Workbook
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import random
import operator
import numpy as np
import datetime
A=np.zeros((38,2366),dtype=np.int)
sm_lncRNA=np.loadtxt('small molecule drug-lncRNA association.txt',dtype=np.int)
for pair in sm_lncRNA:
    A[pair[0]-1][pair[1]-1]=1
    
ls1=np.loadtxt('LncRNA similarity matrix 1.txt')
ls2=np.loadtxt('LncRNA similarity matrix 2.txt')
ls3=np.loadtxt('LncRNA similarity matrix 3.txt')
ds=np.loadtxt('Small molecule drug similarity matrix.txt')

ls=(ls1+ls2+ls3)/3

U_number=[]
for i in range(38):
    for j in range(2366):
        if A[i][j]==0:
            U_number.append([i+1,j+1])


def range2rect(x,y,start=0):
    M=[]
    N=[]
    for i in range(x):
        for j in range(y):
            N.append(start)
        M.append(N)
        N=[]
    return M
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


X_train=[]
T_lable=[]

P=[]
for i in range(2422):
      P.append(sm_lncRNA[i])
U_all=U_number
random_number=random.sample(range(0,len(U_all)),len(P))
U=[]
for i in random_number:
    U.append(U_all[i])


F_V=range2rect(38,2366)
for i in range(38):
    for j in range(2366):
        vector=np.r_[ds[i],ls[j]]    
        F_V[i][j]=vector
   
X_train=[]
T_lable=[]
for i in P:
    X_train.append(F_V[i[0]-1][i[1]-1])
    T_lable.append(1)
for i in U:
    X_train.append(F_V[i[0]-1][i[1]-1])
    T_lable.append(0)

#训练预测模型
rf=RandomForestClassifier(n_estimators=100,max_features=0.2,min_samples_leaf=10)
rf.fit(X_train[:],T_lable[:])    

#预测小分子药物-lncRNA关联分数
sample_all=[]
for i in range(38):
    for j in range(2366):
        if A[i][j]==0:
            sample_all.append(F_V[i][j])
pr=rf.predict_proba(sample_all)[:,1]

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

