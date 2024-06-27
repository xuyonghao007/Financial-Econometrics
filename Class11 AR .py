# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:00:11 2022

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
#https://www.statsmodels.org/stable/user-guide.html#time-series-analysis
#%%
df = pd.read_excel("固定资产月度数据.xlsx")
print(df.head())

#%%
X=df['固定资产投资额'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0],'p-value: %f' % result[1])
#%%
X=df['随机数据'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0],'p-value: %f' % result[1])
#%%
import statsmodels.tsa.stattools as st
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima_model import ARMA

mod = AutoReg(X,lags=2)
res = mod.fit()
print(res.summary())
#%%
#order_analyze = st.arma_order_select_ic(X, max_ar=5, max_ma=5, ic=['aic', 'bic'])
order_analyze = ar_select_order(X, maxlag=5,  ic='bic')
print("model best lags:",mod.ar_lags)
#即最好的model是加入lag1，lag2与lag3

#%%
import pandas_datareader as pdr
import matplotlib.pyplot as plt

data = pdr.get_data_fred("HOUSTNSA", "1959-01-01", "2019-06-01")
housing = data.HOUSTNSA.pct_change().dropna()
# Scale by 100 to get percentages
housing = 100 * housing.asfreq("MS")
fig, ax = plt.subplots()
ax = housing.plot(ax=ax)
#%%
sel = ar_select_order(housing, 13)
sel.ar_lags
res = sel.model.fit()
fig = res.plot_predict(720, 840)
print(res.summary())

#%%
params = res.params
print(params)
