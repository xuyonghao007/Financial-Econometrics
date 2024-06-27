# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:54:24 2022
@author: xuyonghao
"""
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import seaborn as sns
import time
from scipy import stats
import sys
sys_path="d:\\software\\python3\\lib\\site-packages"
sys.path.append(sys_path)
from linearmodels.panel import PanelOLS # 面板回归

chang_path="D:/中南财经政法大学/课程资料 class info/金融计量/参考数据"
os.chdir(chang_path)
#%% 
whole_data=pd.read_csv("whole_data_female.csv")
import statsmodels.formula.api as smf
import statsmodels.api as sm
#help(sm.OLS)
df=whole_data[['stkcd','year','Female_mng_per','TotalSalary','Female','Age','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality',"Female_monitor_per",'Indirect_per']]
df["TotalSalary"] = (df["TotalSalary"]+1).apply(np.log1p)
df["asset"] = (df["asset"]+1).apply(np.log1p)
df=df.dropna()
df.set_index(["stkcd","year"],inplace = True)

     
#%% 
x1 = df[['Female','Age', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality',"Female_board_per",'Indirect_per']]
x1 = sm.add_constant(x1) 
y = df['TotalSalary']

FE_OLS = PanelOLS(y, x1, entity_effects=True, time_effects=False).fit() 
print(FE_OLS)

     
#%% 
from linearmodels import RandomEffects
help(RandomEffects)
     
RE_OLS = RandomEffects(y, x1).fit() 
print(RE_OLS)
