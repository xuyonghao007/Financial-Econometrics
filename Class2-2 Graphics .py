# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 21:56:23 2022

@author: xuyonghao
"""
%clear #清空console

from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import seaborn as sns
import time


plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


chang_path="D:/Python Basics/Class Data/"
os.chdir(chang_path)
Path="D:/Python Basics/Screenshot/"

dataframe_new=pd.read_csv("CEO_Data.csv")
print(dataframe_new.dtypes)

summary_stats=dataframe_new[['Age','TotalSalary','Female','SharEnd', 'TMTP', 'IsDuality']].describe(percentiles=[0.01,0.25,0.75, 0.99])
print(summary_stats)
summary_stats = summary_stats.T ##转置方便看结果
summary_stats=summary_stats.reset_index()
summary_stats.to_csv("summary_stats_CEO.csv", index=False,encoding="utf_8_sig") # 中文简体存储问题



for i in dataframe_new[['Age','TotalSalary','Female','SharEnd', 'TMTP', 'IsDuality']]:
    dataframe_new[i]=np.where(dataframe_new[i].isnull(), np.nan, winsorize(np.ma.masked_invalid(dataframe_new[i]),limits=(0.01,0.01)))
#np.where(condition, x, y),满足condition是x，否则y
#Mask an array where invalid values occur (NaNs or infs).即在winsor是去掉空值

summary_stats2=dataframe_new[['Age','TotalSalary','Female','SharEnd', 'TMTP', 'IsDuality']].describe(percentiles=[0.01,0.25,0.75, 0.99])
summary_stats2 = summary_stats2.T ##转置方便看结果
summary_stats2=summary_stats2.reset_index()
with pd.ExcelWriter('summary_stats_CEO.xlsx') as writer:
     summary_stats.to_excel(writer, sheet_name='summary_before_winsor')
     summary_stats2.to_excel(writer, sheet_name='summary_after_winsor')

##################################################
#%%

TotalSalary=dataframe_new.TotalSalary
#Condition=(dataframe_new.TotalSalary ==0) #这里加不加括号都行，加了括号方便理解月度
#index2 = dataframe_new[Condition].index.tolist()
#TotalSalary=TotalSalary.drop(index2) #删除0的变量
plt.title('直方图')
plt.ylabel('样本频次')
plt.xlabel('薪酬-元')
#plt.hist(TotalSalary,bins=25,alpha=0.7)
###直方图
plt.legend()
plt.hist(TotalSalary,bins=25,range=(TotalSalary.quantile(0.01),TotalSalary.quantile(0.99)),alpha=0.7)
Save_Path=Path+"薪酬直方图.png"
plt.savefig(Save_Path)
plt.show()
#%%
Save_Path=Path+"薪酬密度图.png"
TotalSalary=dataframe_new.TotalSalary
plt.title('密度函数图')
plt.ylabel('样本概率')
plt.xlabel('薪酬-元')
#sns.kdeplot(df['TotalSalary'],shade=True)
sns.kdeplot(TotalSalary,shade=True)
scatter_fig = sns.kdeplot(TotalSalary,shade=True).get_figure()
plt.legend()
scatter_fig.savefig(Save_Path)
plt.show()
#%%

df=dataframe_new

for i in df[['Age','TotalSalary','Female','SharEnd', 'IsDuality']]:
    Save_Path=Path+"Hist_"+i+".png"
    plt.hist(df[i],bins=25,alpha=0.7)
    plt.savefig(Save_Path)
    plt.show()
    plt.title('概率密度图')
    sns.kdeplot(df[i],shade=True)
    plt.show()
    scatter_fig = sns.kdeplot(df[i],shade=True).get_figure()
    Save_Path=Path+"Density_"+i+".png"
    scatter_fig.savefig(Save_Path)

for i in df[['Age','TotalSalary','SharEnd', 'IsDuality']]:
    Save_Path=Path+"Group_Hist"+i+".png"
    plt.title('分组直方图')
    sns.distplot(df[df.Female==1][i],bins=20,kde=False,norm_hist=True,color='hotpink',label='女性')
    sns.distplot(df[df.Female==0][i],bins=20,kde=False,norm_hist=True,color='seagreen',label='男性')
    plt.legend()
    plt.savefig(Save_Path)
    plt.show()
    plt.title('分组概率密度图')
    sns.kdeplot(df[df.Female==0][i],label='女性',shade=True)
    sns.kdeplot(df[df.Female==1][i],label='男性',shade=True)
    plt.legend()
    plt.show()
#%%

#by_gender_data = df[['Age','TotalSalary','SharEnd', 'IsDuality']].groupby(df['female'])
#str_date=str(df['Reptdt'])
#print(type(str_date))
#df['Year'] = df.apply(lambda x : 1 if x['Gender']=="女" else 0,axis=1) 

###################################################
df=dataframe_new[['Age','TotalSalary','SharEnd', 'IsDuality','Reptdt','Female',]]
df['Year'] = df.apply(lambda x : int(x['Reptdt'][0:4]),axis=1) 
#df['Year'] = df.apply(lambda x : time.strptime(x['Year'],"%Y"),axis=1) 
print(df['Year'][0:9])
print(df.head())
#df.columns = df.columns.get_level_values(0)

grouped_way=df.groupby(['Year'])
mean_by_year=grouped_way.mean()
std_by_year=grouped_way.std()
print(mean_by_year)
print(std_by_year)
mean_by_year=mean_by_year.reset_index()
Condition=(mean_by_year.Year <=2006) #这里加不加括号都行，加了括号方便理解月度
index_condition = mean_by_year[Condition].index.tolist()
mean_by_year=mean_by_year.drop(index_condition) #删除2008年以前的
#%%
plt.title('上市公司CEO女性占比')
plt.ylabel('女性CEO比例')
plt.xlabel('时间-年')
plt.bar(mean_by_year.Year,mean_by_year.Female,width=0.7)
plt.xticks(mean_by_year.Year, mean_by_year.Year, rotation=90)
Save_Path=Path+"上市公司CEO女性占比.png"
plt.savefig(Save_Path)
plt.show()
#%%
plt.title('上市公司CEO女性占比-折线图')
plt.ylabel('女性CEO比例')
plt.xlabel('时间-年')
plt.plot(mean_by_year.Year,mean_by_year.Female,color="r")
plt.xticks(mean_by_year.Year, mean_by_year.Year, rotation=90)
Save_Path=Path+"上市公司CEO女性占比-折线图.png"
plt.savefig(Save_Path)
plt.show()
#%%
plt.title('上市公司CEO女性占比-折线与柱状图')
plt.ylabel('女性CEO比例')
plt.xlabel('时间-年')
plt.bar(mean_by_year.Year,mean_by_year.Female,width=0.7)
plt.plot(mean_by_year.Year,mean_by_year.Female,color="r")
plt.xticks(mean_by_year.Year, mean_by_year.Year, rotation=90)
Save_Path=Path+"上市公司CEO女性占比-折线与柱状图.png"
plt.savefig(Save_Path)
plt.show()
#%%
###################################################


grouped_way=df.groupby(['Year','Female'])
mean_by_year=grouped_way.mean()
mean_by_year=mean_by_year.reset_index()

for i in mean_by_year[['Age','TotalSalary','SharEnd', 'IsDuality']]:
    Save_Path=Path+"Group_Hist"+i+".png"
    Title_name='分组柱状图'+i
    plt.xlabel('时间-年')
    plt.ylabel(i)
    plt.title(Title_name, fontsize=10)
    x1=mean_by_year[mean_by_year.Female==1].Year-0.15
    x2=mean_by_year[mean_by_year.Female==0].Year+0.15
    y1=mean_by_year[mean_by_year.Female==1][i]
    y2=mean_by_year[mean_by_year.Female==0][i]
    plt.bar(x1,y1,color='r',width=0.3,label='female')
    plt.bar(x2,y2,color='b',width=0.3,label='male')
    plt.legend()
    plt.grid() #显示网格线
    plt.show()
#%%
###################################################


grouped_way=df.groupby(['Year','Female'])
mean_by_year=grouped_way.mean()
mean_by_year=mean_by_year.reset_index()
Condition=(mean_by_year.Year <=2006) #这里加不加括号都行，加了括号方便理解月度
index_condition = mean_by_year[Condition].index.tolist()
mean_by_year=mean_by_year.drop(index_condition) #删除2008年以前的

for i in mean_by_year[['Age','TotalSalary','SharEnd', 'IsDuality']]:
    Save_Path=Path+"Group_Hist"+i+".png"
    Title_name='分组柱状图'+i
    plt.xlabel('时间-年')
    plt.ylabel(i)
    plt.title(Title_name, fontsize=10)
    x1=mean_by_year[mean_by_year.Female==1].Year
    x2=mean_by_year[mean_by_year.Female==0].Year
    y1=mean_by_year[mean_by_year.Female==1][i]
    y2=mean_by_year[mean_by_year.Female==0][i]
    plt.plot(x1,y1,color='r',label='female')
    plt.plot(x2,y2,color='b',label='male')
    plt.legend()
    plt.grid() #显示网格线
    plt.show()

###################################################

x1=df.Age
y1=df.TotalSalary
plt.scatter(x1,y1,color='r')
plt.show()



