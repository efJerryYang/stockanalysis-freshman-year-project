#!/usr/bin/env python
# coding: utf-8

# In[12]:


'''函数DataProcess作用处理数据，传入参数n，n为0时处理training中的数据，为1时处理training中的数据
n=2处理子窗口要出现的数据（training + trained）
根据股票名和模型名分组，提取最新时间和最高正确率平均正确率，并且存入该目录下的MaxMeanTrainAcc.csv文件'''
import numpy as np
import pandas as pd
import os
# import csv


def DataProcess(n):
    path = os.path.join(os.getcwd(), 'ProjectStorage', 'Log')

    if (n == 0):
        df = pd.read_csv(os.path.join(path, 'training.csv'), encoding='utf-8')
        grouped = df.groupby(by=['Stock Name', 'Model Type', 'Task'])
        MaxMeanData = grouped.agg({'Time': np.max, 'Valid Acc': [np.max, np.mean], 'Div Rate': np.max})
        MaxMeanData.to_csv(os.path.join(path, 'MaxMeanTrainAcc.csv'), encoding='utf-8',
                           index=True)  # 得到股票名，模式名以及对应的正确率最大值Max和平均值Mean最新时间TimeMax，存入此路径下的MaxMeanTrainAcc.csv

    if (n == 1):
        df = pd.read_csv(os.path.join(path, 'trained.csv'), encoding='utf-8')

        grouped = df.groupby(by=['Stock Name', 'Model Type', 'Task'])
        MaxMeanData = grouped.agg({'Time': np.max, 'Valid Acc': [np.max, np.mean]})
        MaxMeanData.to_csv(os.path.join(path, 'MaxMeanTrainAcc.csv'), encoding='utf-8',
                           index=True)  # 得到股票名，模式名以及对应的正确率最大值Max和平均值Mean最新时间TimeMax，存入此路径下的MaxMeanTrainAcc.csv

    if (n == 2):
        df1 = pd.read_csv(os.path.join(path, 'training.csv'), encoding='utf-8')
        df2 = pd.read_csv(os.path.join(path, os.path.join(path, 'trained.csv')), encoding='utf-8')
        MaxMeanData = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
        # MaxMeanData=df1
        # grouped = df_add.groupby(by=['Stock Name', 'Model Type'])
        # MaxMeanData = grouped.agg({'Time': np.max, 'Valid Acc': [np.max, np.mean]})
        MaxMeanData.to_csv(os.path.join(path, 'MaxMeanTrainAcc.csv'), encoding='utf-8',
                           index=True)  # 得到股票名，模式名以及对应的正确率最大值Max和平均值Mean最新时间TimeMax，存入此路径下的MaxMeanTrainAcc.csv
