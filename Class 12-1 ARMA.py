# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:44:11 2022

@author: xuyonghao
"""


import numpy as np
import sys  
sys_path="d:\\software\\python3\\lib\\site-packages"
sys.path.append(sys_path)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMA

#%%
dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
dta.index.freq = dta.index.inferred_freq
del dta["YEAR"]
dta.plot(figsize=(12, 8))
#%%

arma_mod20 = ARIMA(dta, order=(2, 0, 0)).fit()
print(arma_mod20.summary())

print(arma_mod20.aic,arma_mod20.bic)
print(arma_mod20.params)

#%%
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax = arma_mod20.resid.plot(ax=ax)
#%%

print(arma_mod20.predict("2009","2023",dynamic=True))

import statsmodels.tsa.stattools as st


order_analyze = st.arma_order_select_ic(dta, max_ar=8, max_ma=5, ic=['aic'])
print(order_analyze)
print(order_analyze.aic_min_order)