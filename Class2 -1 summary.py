# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:43:22 2022

@author: xuyonghao
"""

import os
import pandas as pd
import numpy as np
chang_path="D:\\Python Basics\\Class Data\\"
os.chdir(chang_path)
new_path=os.getcwd()
print(new_path)


file_name1="D:\\中南财经政法大学\\课程资料 class info\\金融计量\\参考数据\\董监高个人特征文件"+"\\TMT_FIGUREINFO.csv"
dataframe_whole=pd.read_csv(file_name1)
print(dataframe_whole.columns)

 
dataframe_whole.drop(columns = ['IsSupervisor'],inplace = True)
#labels：要删除的行或列，用列表给出
#axis：默认为0，指要删除的是行，删除列时需指定axis为1
#index ：直接指定要删除的行，删除多行可以使用列表作为参数
#columns：直接指定要删除的列，删除多列可以使用列表作为参数
#inplace: 默认为False，该删除操作不改变原数据；inplace = True时，改变原数据

print(dataframe_whole.columns)
print(dataframe_whole.dtypes) #有时候变量名过长，就用循环逐次打印

#for var in dataframe_whole.columns:
    #print(var,dataframe_whole[var].dtypes)
    
print(dataframe_whole.head())
#DataFrame.drop(labels=None,axis=0, index=None, columns=None, inplace=False)
#dataframe_whole.head().to_csv("temp_head.csv", index=False,encoding="utf_8_sig") # 中文简体存储问题

pd.set_option('display.max_rows',20) #设置现实最多20行

new_dataframe=dataframe_whole[['Stkcd', 'Reptdt', 'PersonID', 'Name', 'BirthPlace', 'BirAreaCode', 'Gender', 'Age',
        'Degree', 'Major', 'TotalSalary', 'Allowance', 'SharEnd', 'IsMTMT', 'TMTP', 
        'IsDuality', 'Funback', 'OveseaBack', 'Academic']]

##根据变量选择进行删除
print(new_dataframe['IsMTMT'].value_counts())
print(new_dataframe['TMTP'].value_counts())
#TMTP [高管职务类别] - 1=CEO（包含首席执行官、总经理）， 2=CFO， 3=CEO（包含首席执行官、总经理）和CFO兼任，4=其他。
print(new_dataframe['Gender'].value_counts(),new_dataframe.dtypes)

################
Condition=(new_dataframe.IsMTMT ==0 )#这里加不加括号都行，加了括号方便理解月度
print(Condition)
index1 = new_dataframe[Condition].index.tolist()
print(type(index1))
new_dataframe=new_dataframe.drop(index1) #inplace值设定为True,则原数组内容直接被改变
print(new_dataframe['Gender'].value_counts(),new_dataframe.dtypes)
################
Condition=(new_dataframe.TMTP ==2) | (new_dataframe.TMTP ==4) #这里加不加括号都行，加了括号方便理解阅读
print(Condition)
index2 = new_dataframe[Condition].index.tolist()
new_dataframe=new_dataframe.drop(index2)
print(new_dataframe['Gender'].value_counts())
################


new_dataframe['Female'] = new_dataframe.apply(lambda x : 1 if x['Gender']=="女" else 0,axis=1) 
#方式2，lambda a 其实代表的是df。  方式1中指定了列名，不用加axis=1


new_dataframe.to_csv("CEO_Data.csv", index=False,encoding="utf_8_sig") # 中文简体存储问题


################
summary_stats=new_dataframe[['Age','TotalSalary','Female','SharEnd', 'TMTP', 'IsDuality']].describe()
print(summary_stats)
summary_stats = summary_stats.T ##转置方便看结果
summary_stats=summary_stats.reset_index()

summary_stats.to_csv("summary_stats.csv", index=False,encoding="utf_8_sig") # 中文简体存储问题

################