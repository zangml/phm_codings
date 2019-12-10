# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import lightgbm
from lightgbm import *
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error,median_absolute_error,r2_score
import pandas as pd
import numpy as np
import sys
import json
from sklearn.preprocessing import Imputer
class Result:
    explained_variance_score = 0
    mean_absolute_error = 0
    mean_squared_error = 0
    median_absolute_error =0
    r2_score=0
    featureImportances = []
params = {}
params['n_estimators'] = 100  #树的棵数
params['learning_rate'] = 0.1  #学习率
params['max_depth'] =1    #最大深度
params['random_state'] = None   #最少分裂几个子节点
params['train'] = '/usr/local/data/lab_shouce/lab_shouce_train_reg.csv'
params['test'] = '/usr/local/data/lab_shouce/lab_shouce_test_reg.csv'
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

    dt=pd.read_csv(params['train'])
    train = np.array(dt)
    train = Imputer().fit_transform(train)
    train_y = train[:, -1]
    train_x = train[:, :-1]
    dt=pd.read_csv(params['test'])
    test = np.array(dt)
    test = Imputer().fit_transform(test)
    test_y = test[:, -1]
    test_x = test[:, :-1]

    clf =LGBMRegressor(n_estimators=params['n_estimators'],
                            learning_rate=params['learning_rate'],
                            max_depth=params['max_depth'],
                            random_state=params['random_state']).fit(train_x, train_y)

    predict = clf.predict(test_x)

    res = {}
    res['varianceScore'] = explained_variance_score(test_y,predict,multioutput="uniform_average")
    res['absoluteError'] = mean_absolute_error(test_y,predict,multioutput="uniform_average")
    res['squaredError'] = np.sqrt(mean_squared_error(test_y,predict,multioutput="uniform_average"))
    res['medianSquaredError'] = median_absolute_error(test_y,predict)
    res['r2Score'] = r2_score(test_y,predict,multioutput="uniform_average")
    # res['featureImportances'] = clf.feature_importances_.tolist()
    print(json.dumps(res))
except Exception as e:
    print(e)






