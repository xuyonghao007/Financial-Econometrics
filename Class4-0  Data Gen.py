# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:04:56 2022

@author: xuyonghao
"""

"""
Created on Thu Sep  1 13:54:24 2022

@author: xuyonghao
"""

import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import seaborn as sns
import time
from scipy import stats


chang_path="D:/中南财经政法大学/课程资料 class info/金融计量/参考数据"
os.chdir(chang_path)


#%%
file_name1="D:\\中南财经政法大学\\课程资料 class info\\金融计量\\参考数据\\董监高个人特征文件"+"\\TMT_FIGUREINFO.csv"
dataframe_whole=pd.read_csv(file_name1)
pd.set_option('display.max_rows',20) #设置现实最多20行
new_dataframe=dataframe_whole[['Stkcd', 'Reptdt', 'IsSupervisor', 'Gender']]
Condition=(new_dataframe.IsSupervisor==0 )#这里加不加括号都行，加了括号方便理解，删除非董事会成员
index1 = new_dataframe[Condition].index.tolist()
new_dataframe=new_dataframe.drop(index1) #inplace值设定为True,则原数组内容直接被改变
print(new_dataframe['Gender'].value_counts(),new_dataframe.dtypes)
new_dataframe['Female'] = new_dataframe.apply(lambda x : 1 if x['Gender']=="女" else 0,axis=1) 
new_dataframe['Year'] = new_dataframe.apply(lambda x : int(x['Reptdt'][0:4]),axis=1) 
grouped_way=new_dataframe.groupby(['Stkcd','Year']) #group by这里不是写两行
mean_by_stkcd_year=grouped_way.mean()
mean_by_stkcd_year=mean_by_stkcd_year.reset_index()
mean_by_stkcd_year.rename(columns={"Stkcd": "stkcd", "Year": "year", "Female": "Female_monitor_per"},inplace=True)
mean_by_stkcd_year=mean_by_stkcd_year[['stkcd', 'year', 'Female_monitor_per']]
mean_by_stkcd_year.to_csv("Female_monitor_per.csv", index=False,encoding="utf_8_sig") # 中文简体存储问题
#%%
dataframe_whole=pd.read_csv(file_name1)
new_dataframe=dataframe_whole[['Stkcd', 'Reptdt', 'IsMTMT', 'Gender','TMTP',]]
Condition=(new_dataframe.IsMTMT ==0 )#这里加不加括号都行，加了括号方便理解，删除非董事会成员
index1 = new_dataframe[Condition].index.tolist()
new_dataframe=new_dataframe.drop(index1) #inplace值设定为True,则原数组内容直接被改变
#new_dataframe=new_dataframe.drop(new_dataframe[(new_dataframe.IsMTMT==1]).index.tolist()) #仅保留非ceo的女性成员
Condition=(new_dataframe.TMTP ==1 )#这里加不加括号都行，加了括号方便理解，删除非董事会成员
index2 = new_dataframe[Condition].index.tolist()
new_dataframe=new_dataframe.drop(index2) #inplace值设定为True,则原数组内容直接被改变

new_dataframe['Female'] = new_dataframe.apply(lambda x : 1 if x['Gender']=="女" else 0,axis=1) 
new_dataframe['Year'] = new_dataframe.apply(lambda x : int(x['Reptdt'][0:4]),axis=1) 
grouped_way=new_dataframe.groupby(['Stkcd','Year']) #group by这里不是写两行
mean_by_stkcd_year=grouped_way.mean()
mean_by_stkcd_year=mean_by_stkcd_year.reset_index()
mean_by_stkcd_year.rename(columns={"Stkcd": "stkcd", "Year": "year", "Female": "Female_mng_per"},inplace=True)
mean_by_stkcd_year=mean_by_stkcd_year[['stkcd', 'year', 'Female_mng_per']]
mean_by_stkcd_year.to_csv("Female_mng_per.csv", index=False,encoding="utf_8_sig") # 中文简体存储问题
print(mean_by_stkcd_year.head())

#%%
dataframe1=pd.read_csv("CEO_Data.csv")
dataframe1['year'] = dataframe1.apply(lambda x : int(x['Reptdt'][0:4]),axis=1) 
dataframe1=dataframe1[['Stkcd', 'year', 'Female','Age','TotalSalary', 'IsDuality']]
dataframe1.rename(columns={"Stkcd": "stkcd"},inplace=True)
print(dataframe1.dtypes)
#%%
dataframe2=pd.read_csv("Female_board_per.csv")
dataframe3=pd.read_csv("公司年度价值指标.csv")
dataframe4=pd.read_csv("公司年度金融数据.csv")
dataframe5=pd.read_csv("Female_monitor_per.csv")
dataframe6=pd.read_csv("Female_mng_per.csv")

#%%
whole_data=pd.merge(left=dataframe1,right=dataframe2, on=["stkcd","year"] ,how="inner") #默认是 inner交集，outer并集
whole_data=pd.merge(left=whole_data,right=dataframe3, on=["stkcd","year"] ,how="inner") #默认是 inner交集，outer并集
whole_data=pd.merge(left=whole_data,right=dataframe4, on=["stkcd","year"] ,how="inner") #默认是 inner交集，outer并集
whole_data=pd.merge(left=whole_data,right=dataframe5, on=["stkcd","year"] ,how="inner") #默认是 inner交集，outer并集
whole_data=pd.merge(left=whole_data,right=dataframe6, on=["stkcd","year"] ,how="inner") #默认是 inner交集，outer并集

whole_data=whole_data[(whole_data.year<=2019) & (whole_data.year>=2006)]
print(whole_data.head())
print(whole_data['year'].value_counts(),whole_data.dtypes)

#%%
for i in whole_data[['Female_mng_per','Age','TotalSalary','Female','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset',"Female_monitor_per",'Indirect_per']]:
    whole_data[i]=np.where(whole_data[i].isnull(), np.nan, winsorize(np.ma.masked_invalid(whole_data[i]),limits=(0.01,0.01)))

whole_data.to_csv("whole_data_female.csv", index=False,encoding="utf_8_sig") # 中文简体存储问题