# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:26:14 2022

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
from statsmodels.tsa.api import VAR
 #%%
 
mdata = sm.datasets.macrodata.load_pandas().data
# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
 #%%
from statsmodels.tsa.base.datetools import dates_from_str
quarterly = dates_from_str(quarterly)
mdata = mdata[['realgdp','realcons','realinv']]
mdata.index = pd.DatetimeIndex(quarterly)
 #%%

plt.title('Graph1 original data')
plt.plot(mdata['realgdp'],label='realgdp')
plt.plot(mdata['realcons'],label='realcons')
plt.plot(mdata['realinv'],label='realinv')
plt.legend()
plt.show()

 #%%
data = np.log(mdata).diff().dropna()
 #%%
plt.title('Graph2 Diff data')
plt.plot(data['realgdp'],label='realgdp')
plt.plot(data['realcons'],label='realcons')
plt.plot(data['realinv'],label='realinv')
plt.legend()
plt.show()
#%%
# make a VAR model
model = VAR(data)
results = model.fit(1)
print(results.summary())

#%%
order_results = model.fit(maxlags=15, ic='aic')
print(order_results.summary())

#%%
lag_order = order_results.k_ar
print(results.forecast(data.values[-lag_order:], 5))
#第一个参数是初始的值，第二个参数是未来要预测多少期，
# model是滞后k期的，就是倒数k个数值作为初始值
#%%
results.plot_forecast(100)
#直接把预测的结果以图像形式画出
#%%
irf = results.irf(5)
irf.plot()

#%%
irf.plot(impulse='realgdp')

#%%
irf.plot_cum_effects(impulse='realgdp')
#%%
test1 = model.fit(1).test_causality('realgdp', causing=['realcons'], kind='wald', signif=0.05)
print(test1,test1.pvalue)
#%%
test2 = model.fit(1).test_causality('realcons', causing=['realgdp'], kind='wald', signif=0.05)
print(test2,test2.pvalue)

#%%
test3 =results.test_causality('realgdp', ['realinv', 'realcons'], kind='wald', signif=0.05)
print(test3,test3.pvalue)
#%%
fevd = results.fevd(5)
print(fevd.summary())

#%%
.,
results.fevd(10).plot()
