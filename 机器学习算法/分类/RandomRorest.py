# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
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
params['n_estimators'] = 10
params['max_depth'] = None
params['max_features'] = "auto"
params['min_samples_split'] = 2
params['min_samples_leaf'] = 1
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

    train = np.array(pd.read_csv(params['train']))
    train_y = train[:, -1]
    train_x = train[:, :-1]

    test = np.array(pd.read_csv(params['test']))
    test_y = test[:, -1]
    test_x = test[:, :-1]

    clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                 max_features=params['max_features'],
                                 max_depth=params['max_depth'],
                                 min_samples_split=params['min_samples_split'],
                                 min_samples_leaf=params['min_samples_leaf']).fit(train_x, train_y)

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
    res['featureImportances'] = clf.feature_importances_.tolist()
    print(json.dumps(res))
except Exception as e:
    print(e)