# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:54:24 2022

@author: xuyonghao
"""

import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import seaborn as sns
from scipy import stats

chang_path="D:/中南财经政法大学/课程资料 class info/金融计量/参考数据"
os.chdir(chang_path)


#%% 生成女性董事会成员比例指标
dataframe_whole=pd.read_csv("D:\\中南财经政法大学\\课程资料 class info\\金融计量\\参考数据\\董监高个人特征文件"+"\\TMT_FIGUREINFO.csv")

pd.set_option('display.max_rows',20) #设置现实最多20行
new_dataframe=dataframe_whole[['Stkcd', 'Reptdt', 'IsMTB', 'IsIdirecotr', 'Gender']]
print(new_dataframe.head())
#%%
new_dataframe=new_dataframe.query("IsMTB ==  1")
#%%
new_dataframe['Female'] = new_dataframe.apply(lambda x : 1 if x['Gender']=="女" else 0,axis=1) 
new_dataframe['Year'] = new_dataframe.apply(lambda x : int(x['Reptdt'][0:4]),axis=1) 
#%%

grouped_way=new_dataframe.groupby(['Stkcd','Year']) #group by这里不是写两行
mean_by_stkcd_year=grouped_way.mean()
#%%
mean_by_stkcd_year=mean_by_stkcd_year.reset_index()

#%%
mean_by_stkcd_year.rename(columns={"Stkcd": "stkcd", "Year": "year", "Female": "Female_board_per"},inplace=True)
mean_by_stkcd_year.rename(columns={"IsIdirecotr": "Indirect_per"},inplace=True)
#%%
mean_by_stkcd_year=mean_by_stkcd_year[['stkcd', 'year', 'Female_board_per',"Indirect_per"]]
mean_by_stkcd_year.to_csv("Female_board_per.csv", index=False,encoding="utf_8_sig") # 中文简体存储问题



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
print(dataframe2.head())
print(dataframe3.head())
print(dataframe4.head())

#%%
whole_data=pd.merge(left=dataframe1,right=dataframe2, on=["stkcd","year"] ,how="inner") #默认是 inner交集，outer并集
whole_data=pd.merge(left=whole_data,right=dataframe3, on=["stkcd","year"] ,how="inner") #默认是 inner交集，outer并集
whole_data=pd.merge(left=whole_data,right=dataframe4, on=["stkcd","year"] ,how="inner") #默认是 inner交集，outer并集
print(whole_data.head())

#%%
whole_data=whole_data[(whole_data.year<=2019) & (whole_data.year>=2006)]
print(whole_data.head())
print(whole_data['year'].value_counts(),whole_data.dtypes)

#%%
for i in whole_data[['Age','TotalSalary','Female','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset']]:
    whole_data[i]=np.where(whole_data[i].isnull(), np.nan, winsorize(np.ma.masked_invalid(whole_data[i]),limits=(0.01,0.01)))
    
    
#%% 
df=whole_data[['Age','TotalSalary','Female','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset']]
df["TotalSalary"] = df["TotalSalary"].apply(np.log1p)
summary_stats=df[['Age','TotalSalary','Female','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset']].describe(percentiles=[0.01,0.25,0.75, 0.99])
print(summary_stats)
summary_stats = summary_stats.T ##转置方便看结果
print(summary_stats)


#%% 
Female_data=df[(df.Female==1)][['Age','TotalSalary','Female','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset']]
Male_data=df[(df.Female==0)][['Age','TotalSalary','Female','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset']]


#%% 
summary_stats1=Female_data.describe(percentiles=[0.01,0.25,0.75, 0.99])
summary_stats1 = summary_stats1.T ##转置方便看结果
summary_stats2= Male_data.describe(percentiles=[0.01,0.25,0.75, 0.99])
summary_stats2 = summary_stats2.T ##转置方便看结果
#summary_stats2=summary_stats2.reset_index()

#%% 

with pd.ExcelWriter('summary_stats_CEO_by_gender.xlsx') as writer:
     summary_stats.to_excel(writer, sheet_name='summary_whole')
     summary_stats1.to_excel(writer, sheet_name='summary_female')
     summary_stats2.to_excel(writer, sheet_name='summary_male')
#%% 

Correlation_stats = df.corr(method='spearman')
print(Correlation_stats,type(Correlation_stats))

from scipy.stats import pearsonr,spearmanr
def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = spearmanr(df[col],df[col2],nan_policy='omit')
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
###定义当输入包含 nan 时如何处理。以下选项可用(默认为‘propagate’)：
#‘propagate’：返回 nan
#‘raise’：抛出错误
#‘omit’：执行忽略 nan 值的计算
p_values = corr_sig(df)
mask = np.invert(np.tril(p_values<0.05))
print(p_values)
print(mask)
print(type(p_values),type(mask))
p_values=pd.DataFrame(p_values)
mask=pd.DataFrame(mask)


with pd.ExcelWriter('Correlation_stats_CEO.xlsx') as writer:
     Correlation_stats.to_excel(writer, sheet_name='Correlation_stats')
     p_values.to_excel(writer, sheet_name='P_value')
     mask.to_excel(writer, sheet_name='Sig_at_5%')
#%% 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

df=whole_data[['TotalSalary','Female','Age','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset']]
df["TotalSalary"] = (df["TotalSalary"]+1).apply(np.log1p)
df["asset"] = (df["asset"]+1).apply(np.log1p)
df=df.dropna()
#df['constant']=1
df=add_constant(df)
#%% 

vif = variance_inflation_factor(df.values,1)
vif=pd.Series([variance_inflation_factor(df.values, i) for i in range(df.shape[1])], index=df.columns)
print("VIF的结果如下\n",vif)



#vif["features"] = df.columns
#print("变量个数",df.shape[1])
#%% 
import statsmodels.formula.api as smf
import statsmodels.api as sm
#help(sm.OLS)
df=whole_data[['TotalSalary','Female','Age','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality']]
df["TotalSalary"] = (df["TotalSalary"]+1).apply(np.log1p)
df["asset"] = (df["asset"]+1).apply(np.log1p)
df=df.dropna()

#Controls = np.column_stack((df['Female']))
controls=df[['Female','Age','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset']]
y = df['TotalSalary']
#Controls = sm.add_constant(Controls) # 添加常数项

print(controls.shape,y.shape)
#%% 
#x = df.reindex(['Female','Age', 'tobinq', 'asset', 'roa', 'debt_asset'], axis="columns")
x1 = df[['Female','Age', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality']]
x1 = sm.add_constant(x1) 
y = df['TotalSalary']
#%% 
result = sm.OLS(y, x1, missing="drop").fit()
print(result.summary())
#%% 


df_result = pd.concat((result.params, result.tvalues), axis=1)
df_result=df_result.rename(columns={0: 'beta', 1: 't'})

with pd.ExcelWriter('OLS_Result_CEO.xlsx') as writer:
     df_result.to_excel(writer, sheet_name='sheet1')

     
#%% 
x2 = df[['Female','Female_board_per','Age', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality']]
#x2['age2']=x2['Age']*x2['Age']

x2 = sm.add_constant(x2) 
y = df['TotalSalary']

result = sm.OLS(y, x2, missing="drop").fit()
print(result.summary())






