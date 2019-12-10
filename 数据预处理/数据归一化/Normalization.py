# -*- coding: utf-8 -*-
from sklearn import preprocessing 
import pandas as pd
import numpy as np
import sys
import csv

params = {}
params['feature_range'] = (0,1)
params['path'] = '/usr/local/data/lab_shouce/dj_nor.csv'
params['opath'] ='/usr/local/data/lab_shouce/dj_nor_out.csv'
argvs = sys.argv
try:
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])
    with open(params['path'],'r') as f:
    #1.创建阅读器对象
        reader = csv.reader(f)
    #2.读取文件第一行数据
        head_row=next(reader)
    data_attribute = []
    for item in head_row:
        data_attribute.append(item)

    #读取数据并删除最后一列标签
    tn = pd.read_csv(params['path']) 
    tn.dropna(inplace=True)
    train = np.array(tn)
    train_x = train[:, :-1]
    
    
    train_y = train[:,-1]
    train_y = np.array(train_y)
   # print(train_y)
    
    #对所有数据行进行标准化
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=params['feature_range'])

    X_train_minmax = min_max_scaler.fit_transform(train_x)
   
    out=np.column_stack((X_train_minmax,train_y))
   
    csvfile2 = open(params['opath'],'w')
    writer = csv.writer(csvfile2)
    writer.writerow(data_attribute)   #存属性
    m = len(out)
    #print(m)
    for i in range(m):
        writer.writerow(out[i])
    
except Exception as e:
    print(e)
    
