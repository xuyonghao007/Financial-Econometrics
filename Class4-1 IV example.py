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
print(sys.path) 

chang_path="D:/中南财经政法大学/课程资料 class info/金融计量/参考数据"
os.chdir(chang_path)



#%% 
whole_data=pd.read_csv("whole_data_female.csv")

import statsmodels.formula.api as smf
import statsmodels.api as sm
#help(sm.OLS)
df=whole_data[['Female_mng_per','TotalSalary','Female','Age','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality',"Female_monitor_per",'Indirect_per']]
df["TotalSalary"] = (df["TotalSalary"]+1).apply(np.log1p)
df["asset"] = (df["asset"]+1).apply(np.log1p)
df=df.dropna()

x1 = df[['Female','Age', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality',"Female_board_per",'Indirect_per']]
x1 = sm.add_constant(x1) 
y = df['TotalSalary']

result = sm.OLS(y, x1, missing="drop").fit()
print(result.summary())


     
#%% 
IV= df[['Female_mng_per','Female_monitor_per','Indirect_per','Female_board_per','Age', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality']]
#x2['age2']=x2['Age']*x2['Age']
IV = sm.add_constant(IV) 
y_female = df['Female']
result = sm.OLS(y_female, IV, missing="drop").fit()
print(result.summary())



#%% 
#import sys  
#print(sys.path)  #查看sys路径

sys_path="d:\\software\\python3\\lib\\site-packages"
sys.path.append(sys_path)

from statsmodels.sandbox.regression.gmm import IV2SLS
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels.iv.model as lm

result_IV = IV2SLS(y,
                  x1, #所有原始自变量
                  IV).fit() #工具变量，还包含除去内生的自变量以外的全部自变量           

#print(result_IV.summary())


#%% 
exog_var = df[['Age', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality',"Female_board_per",'Indirect_per']]

iv_var=df[['Female_mng_per','Female_monitor_per']]
#iv_var=df[['Female_monitor_per']]
mlr2 = lm.IV2SLS(dependent=y, 
                 exog=exog_var, 
                 endog=df["Female"], 
                 instruments=iv_var).fit()
#mlr2.summary.as_csv()

#print(mlr2.summary())
print(mlr2)
print(mlr2.first_stage)

#help(mlr2)
print(mlr2.wu_hausman())
#wu_hausman test需要显著才可以通过，表明

print(mlr2.sargan)
#sargan test需要不显著才可以通过，表明不存在过度识别问题，即外生性满足。注意，是至少需要sargan test通过；但即使通过，并不意味着外生性就满足了


