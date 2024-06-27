# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:54:24 2022

@author: xuyonghao
"""

from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import seaborn as sns
import time
from scipy import stats


chang_path="D:/Python Basics/Class Data/"
os.chdir(chang_path)
Path="D:/Python Basics/Screenshot/"

dataframe_new=pd.read_csv("CEO_Data.csv")
print(dataframe_new.dtypes)



for i in dataframe_new[['Age','TotalSalary','Female','SharEnd', 'IsDuality']]:
    dataframe_new[i]=np.where(dataframe_new[i].isnull(), np.nan, winsorize(np.ma.masked_invalid(dataframe_new[i]),limits=(0.01,0.01)))
dataframe_new['Year'] = dataframe_new.apply(lambda x : int(x['Reptdt'][0:4]),axis=1) 

df=dataframe_new

Female_data=df[(df.Female==1) & (df.Year<=2008) & (df.Year>=2000)][['Age','TotalSalary','SharEnd', 'TMTP', 'IsDuality']]
Male_data=df[ (df.Female==0)  & (df.Year<=2008) & (df.Year>=2000)][['Age','TotalSalary','SharEnd', 'TMTP', 'IsDuality']]

summary_stats1=Female_data.describe(percentiles=[0.01,0.25,0.75, 0.99])
summary_stats1 = summary_stats1.T ##转置方便看结果
summary_stats2= Male_data.describe(percentiles=[0.01,0.25,0.75, 0.99])
summary_stats2 = summary_stats2.T ##转置方便看结果
summary_stats2=summary_stats2.reset_index()
with pd.ExcelWriter('summary_stats_CEO_by_gender.xlsx') as writer:
     summary_stats1.to_excel(writer, sheet_name='summary_female')
     summary_stats2.to_excel(writer, sheet_name='summary_male')
     
print(summary_stats1) 
print(summary_stats2) 

Female_age=Female_data['Age'].values
Male_age=Male_data['Age'].values

diff_age=Female_data['Age'].mean(skipna=True)-Male_data['Age'].mean(skipna=True)

t2, p2 = stats.ttest_ind(Male_age, Female_age,nan_policy='omit')
print("女性比男性CEO年龄大:",diff_age ,"t:值",t2, "p值:",p2)

diff_salary=Female_data['TotalSalary'].mean(skipna=True)-Male_data['TotalSalary'].mean(skipna=True)

t3, p3 = stats.ttest_ind(Female_data['TotalSalary'].values, Male_data['TotalSalary'].values,nan_policy='omit')
print("女性比男性CEO薪酬高:",diff_salary ,"t:值",t3, "p值:",p3)





