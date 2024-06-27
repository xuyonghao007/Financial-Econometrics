# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:06:25 2022

@author: xuyonghao
"""

import numpy as np
import sys  
sys_path="d:\\software\\python3\\lib\\site-packages"
sys.path.append(sys_path)
import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller
chang_path="D:/中南财经政法大学/课程资料 class info/金融计量/参考数据"
os.chdir(chang_path)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMA
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
#%%

df = pd.read_excel("固定资产月度数据.xlsx")
print(df.head())
del df["随机数据"]
df.set_index("时间").plot()

#%%
X=df['固定资产投资额'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0],'p-value: %f' % result[1])
#%%
df['log固定资产投资额']=np.log(df['固定资产投资额'])
X=df['log固定资产投资额'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0],'p-value: %f' % result[1])


#%%
df['diff_log固定资产投资额']=df['log固定资产投资额'].diff()
print(df.head())
#%%
X=df['diff_log固定资产投资额'].dropna().values
df.set_index("时间").plot(df['diff_log固定资产投资额'])
#%%
result = adfuller(X)
print('ADF Statistic: %f' % result[0],'p-value: %f' % result[1])
#%%
df = np.log(X).diff().dropna()

arma_mod20 = ARIMA(dta, order=(2, 0, 0)).fit()

#%%
X=df['固定资产投资额'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0],'p-value: %f' % result[1])