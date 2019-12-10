# -*- coding: utf-8 -*-
from sklearn import svm
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import json

class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []
params = {}
params['C'] = 1.0
params['gamma'] ='auto'
params['kernel'] ='rbf'
params['degree'] = 3
params['coef0'] = 0.0
params['tol'] = 0.001
params['train'] = '/usr/local/data/lab_shouce/lab_shouce_train_cla.csv'
params['test'] = '/usr/local/data/lab_shouce/lab_shouce_test_cla.csv'
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

    tn=pd.read_csv(params['train'])
    tn.dropna(inplace=True)
    train = np.array(tn)
    train_y = train[:, -1]
    train_x = train[:, :-1]
    
    tt=pd.read_csv(params['test']) 
    tt.dropna(inplace=True)
    test = np.array(tt)
    test_y = test[:, -1]
    test_x = test[:, :-1]

    clf = svm.SVC(gamma=params['gamma'],
                  C=params['C'],
                  kernel=params['kernel'],
                  degree=params['degree'],
                  coef0=params['coef0'],
                  tol=params['tol']).fit(train_x, train_y)

    predict = clf.predict(test_x)
    precision = precision_score(test_y, predict)
    recall = recall_score(test_y, predict)
    accuracy = accuracy_score(test_y, predict)
    res = {}
    res['precision'] = precision
    res['recall'] = recall
    res['accuracy'] = accuracy
    res['fMeasure'] = f1_score(test_y, predict)
    res['rocArea'] = roc_auc_score(test_y, predict)
    print(json.dumps(res))
except Exception as e:
    print(e)