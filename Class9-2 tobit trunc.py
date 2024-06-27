# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:30:10 2022

@author: xuyonghao
"""

import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import time
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices
import sys  

sys_path="d:\\software\\python3\\lib\\site-packages"
sys.path.append(sys_path)

from py4etrics.truncreg import Truncreg
from py4etrics.tobit import Tobit
#from py4etrics.heckit import Heckit
#from py4etrics.hetero_test import het_test_probit
#print(sys.path) 
chang_path="D:/中南财经政法大学/课程资料 class info/金融计量/参考数据"
os.chdir(chang_path)

# https://py4etrics-github-io.translate.goog/21_TruncregTobitHeckit.html?_x_tr_sl=ja&_x_tr_tl=en&_x_tr_hl=zh-CN

#%% probit model and margin effect

whole_data=pd.read_csv("whole_data_female.csv")
df=whole_data[['Female_mng_per','TotalSalary','Female','Age','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality',"Female_monitor_per",'Indirect_per']]
df["TotalSalary"] = (df["TotalSalary"]+1).apply(np.log1p)
df["asset"] = (df["asset"]+1).apply(np.log1p)
df=df.dropna()

x1 = df[[ 'asset', 'roa', 'debt_asset',"Female_board_per",'Indirect_per']]
x1 = sm.add_constant(x1) 
y = df['Female_mng_per']

result = sm.OLS(y, x1, missing="drop").fit()
print(result.summary())

#%% probit model and margin effect

left = 0
censor = df['Female_mng_per'].apply(lambda x: -1 if x==left else 0)

formula = 'Female_mng_per ~ asset + roa + debt_asset+ Female_board_per+ Indirect_per'

res_tobit = Tobit.from_formula(formula,cens=censor,left=0,data=df).fit()
print(res_tobit.summary())


#%% probit model and margin effect
#print(df.describe()) print(df_drop0.describe())
df_drop0 = df.query('Female_mng_per > 0')  #仅仅取出df中Female_mng_per大于0的，产生截尾数据
formula = 'Female_mng_per ~ asset + roa + debt_asset+ Female_board_per+ Indirect_per'

res_trunc = Truncreg.from_formula(formula,left=0,data=df_drop0).fit()
print(res_trunc.summary())
