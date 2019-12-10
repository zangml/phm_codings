# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:35:27 2018

@author: zangmenglei
"""

import pandas as pd
import numpy as np
import sys
import csv
from scipy import signal

argvs = sys.argv

params = {}
params['path'] = '/usr/local/data/lab_shouce/djms_fea.csv'
params['opath'] ='/usr/local/data/lab_shouce/djms_fea_out.csv'
params['len_piece']=20   
params['min_fre']=1600
params['max_fre']=2400
params['freq']=25600

def  one_row(arr,label_mean):
      result_list = []
      arr_add=arr.loc[:,:]
      for j in range(int(255/params['len_piece'])):     
             arr_add =arr_add.append(arr,ignore_index=True)
      for i in arr_add.columns:
             flist,plist = signal.welch(arr_add[i],params['freq'])     
             main_ener = np.square(plist[np.logical_and(flist>=params['min_fre'],flist<params['max_fre'])]).sum()
             #main_ener2 = np.square(plist[np.logical_and(flist>=3600,flist<3950)]).sum()
             #ratio = main_ener1/main_ener2      
             list_para = [main_ener]
             result_list.extend(list_para)
      list_label=[label_mean.mean()]
      result_list.extend(list_label)
      return result_list

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
    #1.´´½¨ÔÄ¶ÁÆ÷¶ÔÏó
        reader = csv.reader(f)
    #2.¶ÁÈ¡ÎÄ¼þµÚÒ»ÐÐÊý¾Ý
        head_row=next(reader)
        head_label=head_row[-1]
        head_row=head_row[:-1]      
        head_label2=head_row[-1]
    data_head=[]
    for i in head_row:     
         head_out=[]
         #head_out=[i+'_main_ener1',i+'_main_ener2',i+'_ratio']
         head_out=[i+'_main_ener']
         data_head.extend(head_out)
    headlabel_out=[head_label]
    data_head.extend(headlabel_out)
    data_out=[]
    data_out.append(data_head)
    df = pd.read_csv(params['path'])
    data=np.array(df)
    lenth=data[:,-1]
    lenth=len(np.array(lenth))
    i=1
    while i*params['len_piece']<lenth:           #¼Ó´°½øÐÐÌØÕ÷ÌáÈ¡    
        arr_pic=df.loc[(i-1)*params['len_piece']:i*params['len_piece']-1,:head_label2]
        label_pic=df.loc[(i-1)*params['len_piece']:i*params['len_piece']-1,head_label]       
        i=i+1
        data_out.append(one_row(arr_pic,label_pic))
    if params['len_piece']>lenth:                 #Èç¹û´°´óÐ¡´óÓÚ×ÜÊý¾ÝÁ¿Ôò¶ÔÈ«Êý¾Ý½øÐÐÌØÕ÷ÌáÈ¡ 
        arr_pic=df.loc[:lenth,:head_label2]
        label_pic=df.loc[:lenth,head_label]
    else:                                                        #×îºóÊ£ÏÂµÄÊý¾ÝÓëÇ°×éÖØµþ´Õ×ã´°´óÐ¡½øÐÐÌØÕ÷ÌáÈ¡
        arr_pic=df.loc[lenth-params['len_piece']:lenth,:head_label2]
        label_pic=df.loc[lenth-params['len_piece']:lenth,head_label]
    data_out.append(one_row(arr_pic,label_pic))
    wrtocsv = pd.DataFrame(data_out)
    wrtocsv.to_csv(params['opath'],index=False,header=False)
  
except Exception as e:
    print(e)