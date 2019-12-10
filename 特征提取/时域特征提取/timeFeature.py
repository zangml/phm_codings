# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:35:27 2018

@author: zangmenglei
"""

import pandas as pd
import numpy as np
import sys
import csv
from scipy import stats

argvs = sys.argv

params = {}
params['path'] = '/usr/local/data/lab_shouce/djms.csv'
params['opath'] ='/usr/local/data/lab_shouce/djms_out.csv'
params['len_piece']=10 #窗口长度

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
    for item in head_row[0:-1]:
        data_attribute.append(item+'_avg')
        data_attribute.append(item+'_std')
        data_attribute.append(item+'_var')
        data_attribute.append(item+'_skew')
        data_attribute.append(item+'_kur')
        data_attribute.append(item+'_ptp')
        
    # data_attribute.append(head_row[-1])
    #print (data_attribute)
    #df = pd.read_csv(params['path'])
    data_attribute.append(head_row[-1])
    df1 = pd.DataFrame(pd.read_csv(params['path']))
    #lable=df1.iloc[:,-1:].values
    #print(lable[10])
    tn = pd.read_csv(params['path']) 
    tn.dropna(inplace=True)
    train = np.array(tn)
    train_x = train[:, :]


#存标签
    train_y = train[:,-1]
    #print(train_y[5])
    
    df_data = pd.DataFrame(pd.read_csv(params['path'])).iloc[:,:-1]
    sum=len(df_data)
    #print(sum)
    result_out=[]
    #print(sum)
    for i in range (0,sum,params['len_piece']):
        df=df_data[i:i+params['len_piece']]
        result_list = []
        for i in df.columns:#每一列
            list_para = [df[i].mean(),df[i].std(),np.var(df[i]),stats.skew(df[i]),stats.kurtosis(df[i]),df[i].ptp()]
            result_list.extend(list_para)
        #result_list.extend(list_lable)
        result_out.append(result_list)
    out=np.column_stack((result_out,train_y[:len(result_out)]))
    csvfile2 = open(params['opath'],'w')
    writer = csv.writer(csvfile2)
    writer.writerow(data_attribute)   #存属性
    m = len(out)
    print(m)
    for i in range(m):
        writer.writerow(out[i])  #存数据
except Exception as e:
    print(e)