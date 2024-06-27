# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:53:28 2022

@author: xuyonghao
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import sys  
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices
sys_path="d:\\software\\python3\\lib\\site-packages"
sys.path.append(sys_path)

#print(sys.path) 
chang_path="D:/中南财经政法大学/课程资料 class info/金融计量/参考数据"
os.chdir(chang_path)
# https://www.statsmodels.org/dev/index.html
#%% 


whole_data=pd.read_csv("whole_data_female.csv")
df=whole_data[['Female_mng_per','TotalSalary','Female','Age','Female_board_per', 'tobinq', 'asset', 'roa', 'debt_asset', 'IsDuality',"Female_monitor_per",'Indirect_per']]
df["TotalSalary"] = (df["TotalSalary"]+1).apply(np.log1p)
df["asset"] = (df["asset"]+1).apply(np.log1p)
df=df.dropna()
#%% 
x1 = df[[ 'asset', 'roa', 'debt_asset',"Female_board_per",'Indirect_per']]
x1 = sm.add_constant(x1) 
y = df['Female']

result = sm.OLS(y, x1, missing="drop").fit()
print(result.summary())

#%% logit model

y,X = dmatrices('Female ~ asset + roa + debt_asset+ Female_board_per+ Indirect_per',data = df,return_type='dataframe')
#patsy.dmatrices('y ~ x0 + x1 + 0', data) 默认有截距项，加上0表示删去截距
logit = sm.Logit(y,X)
results = logit.fit()
print(results.summary())
#%% margin effect 
#get_margeff(at='overall', method='dydx', atexog=None, dummy=False, count=False)
#‘overall’, 平均边际效应,默认. ‘mean’, 样本均值处的边际效应. ‘median’, 样本中值处的边际效应.
#method 'dydx’ - dy/dx， ‘eyex’ - d(lny)/d(lnx) ，‘dyex’ - dy/d(lnx) ，‘eydx’ - d(lny)/dx
margeff = results.get_margeff()
print(margeff.summary())
#%% margin effect 
Xtest = X
ytest = y
  
# performing predictions on the test datdaset
yhat = results.predict(Xtest)
prediction = list(map(round, yhat))  ## round取整数
  
# comparing original and predicted values of y
print('Actual values', list(ytest.values)[0:100])
print('Predictions :', prediction[0:100])
#%% margin effect 
yhat = results.predict(X.iloc[1000])
print(yhat)


#%% probit model and margin effect
probit_model = sm.Probit(y,X)
results = probit_model.fit()
print(results.summary())
margeff = results.get_margeff()
print(margeff.summary())



