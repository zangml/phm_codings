
# coding: utf-8
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
import json

import warnings
warnings.filterwarnings('ignore')
params = {}
params['learning_rate'] = 0.1  #学习率
params['weight_decay'] =1e-8 
params['iterations'] =200 #迭代次数
params['train'] = '/usr/local/data/djms/train.csv'
params['test'] = '/usr/local/data/djms/test.csv'

try:
    
    argvs = sys.argv

    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])




    data1 = pd.DataFrame(pd.read_csv(params['train']))
    data2 = pd.DataFrame(pd.read_csv(params['test']))

    input_col_size=len(data1.columns)-1


    X1 = data1.drop(['label'], axis=1) 
    X2 = data2.drop(['label'], axis=1) 


    Y1 = data1.loc[:,['label']]
    Y2 = data2.loc[:,['label']]


# In[4]:


    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler()
    X1_scaler =  X_scaler.fit_transform(X1)
    X2_scaler =  X_scaler.fit_transform(X2)


# # 滑窗

# In[5]:


#滑窗，处理成3d数据
    def window_data(dat):
        window = 10
        win = np.zeros((dat.shape[0]-window+1, window, dat.shape[1]))
        rn = dat.shape[0]-window+1
        for i in range(rn):
            win[i] = dat[i:i+window,:]
        return win


# In[6]:


    window_x_train1 = torch.from_numpy(window_data(X1_scaler))
    window_x_train2 = torch.from_numpy(window_data(X2_scaler))


# In[7]:


#单向lstm
    window_y_train1 = torch.from_numpy(Y1[9:].values) #window-1
    window_y_train2 = torch.from_numpy(Y2[9:].values)


# In[8]:


    train1_dataset = Data.TensorDataset(window_x_train1,window_y_train1)
    train2_dataset = Data.TensorDataset(window_x_train2,window_y_train2)

    train1_loader = DataLoader(train1_dataset, batch_size=512, shuffle=False)
    train2_loader = DataLoader(train2_dataset, batch_size=512, shuffle=False)

# In[21]:


    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
        #self.input = nn.Linear(28,128)
            self.rnn = nn.LSTM(
                input_size=input_col_size,
                hidden_size=input_col_size,
                num_layers=1,
            # batch_first=False  # (time_step,batch,input)
                batch_first = False,   # (batch,time_step,input)
                bidirectional=False,  
        )
            self.out = nn.Sequential(
                nn.Linear(int(input_col_size),int(input_col_size/2)),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(int(input_col_size/2), 1),
                nn.ReLU(),
        )
        def window(self, dat):
            window = 10
        #win = torch.from_numpy(np.zeros((dat.shape[0]-window+1, window, dat.shape[1]))).cuda()
            win = Variable(torch.zeros(dat.shape[0]-window+1,window,dat.shape[1])).cuda()
            rn = dat.shape[0]-window+1
            for i in range(rn):
                win[i] = dat[i:i+window,:]
            return win
        def forward(self,x):
            r_out, (h_n, h_c) = self.rnn(x, None)
            out = self.out(r_out[:,-1,:]) #切片-1代表倒数第一索引，即取lstm 最后一个cell的输出
            return out#,encode1,encode2
# In[22]:


    model1 = RNN()

    if torch.cuda.is_available():
        model1 = model1.cuda()
    optimizer = torch.optim.Adam(model1.parameters(), lr=params['learning_rate'],weight_decay=params['weight_decay']) #优化器
    criterion = nn.MSELoss() 
    criterion2 = nn.MSELoss()
    criterion_mmd = nn.L1Loss()


# In[23]:


    y_loss1 = []
    for epoch in range(params['iterations']):
        i = 0
        for data1 in train1_loader:
            i=i+1
            batch_x1,batch_y1 = data1
            if torch.cuda.is_available():
                batch_x1 = batch_x1.cuda()
                batch_y1 = batch_y1.cuda()
            else:
                batch_x1 = Variable(batch_x1)
                batch_y1 = Variable(batch_y1)
            predict1 = model1(batch_x1.float())
            loss = criterion(predict1, batch_y1[:,:].float())#+0.8*loss_func(Sae1,Sae3)
            print_loss = loss.data.item()
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
        y_loss1.append(print_loss)


# In[25]:


    model1.eval()


# In[26]:


    window_x_train2 = window_x_train2.cuda()
    out1 = model1(window_x_train2.float())
    yp = out1.cpu().detach().numpy()


# In[29]:


#评价分数
    yt = Y2[9:].values

    from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error,median_absolute_error,r2_score
    res = {}
    res['varianceScore'] = explained_variance_score(yt/60,yp/60)
    res['absoluteError'] = mean_absolute_error(yt/60,yp/60)
    res['squaredError']= mean_squared_error(yt/60,yp/60)
    res['medianSquaredError'] = median_absolute_error(yt/60,yp/60)
    res['r2Score'] = r2_score(yt/60,yp/60)

    print(json.dumps(res))
except Exception as e:
    print(e)


