# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:12:45 2022

@author: xuyonghao
"""


import sys  
sys_path="d:\\software\\python3\\lib\\site-packages"
sys.path.append(sys_path)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn import linear_model
import os
#%matplotlib inline 这个是在jupyternotebook里才可以用的
pd.set_option("display.max_columns", 6)
#style.use("fivethirtyeight")

chang_path="D:/中南财经政法大学/课程资料 class info/金融计量/python file/python-causality-handbook-master/causal-inference-for-the-brave-and-true"
os.chdir(chang_path)
#%matplotlib inline


#%%


cigar = (pd.read_csv("data/smoking.csv").drop(columns=["lnincome","beer", "age15to24"]))

print(cigar.query("california ==  True").head()) #query取用命令


#%%
#%matplotlib inline

ax = plt.subplot(1, 1, 1)

(cigar.assign(california = np.where(cigar["california"], "California", "Other States")).groupby(["year", "california"])["cigsale"]
 .mean()
 .reset_index()
 .pivot("year", "california", "cigsale")
 .plot(ax=ax, figsize=(10,5)))

plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Cigarette Sales Trend")
plt.title("Gap in per-capita cigarette sales (in packs)")
plt.legend()
#%%

features = ["cigsale", "retprice"]
#pivot函数用来重塑数据，官方定义如下所示 pivot(index=None, columns=None, values=None)，设定新的索引
inverted = (cigar.query("~after_treatment") # filter pre-intervention period
            .pivot(index='state', columns="year")[features] # make one column per year and one row per state
            .T) # flip the table to have one column per state

print(type(inverted),inverted.shape,inverted.head())

#%%
#Now, we can define our Y variable as the state of California and the X as the other states. #state等于3的时候就是cal
y = inverted[3].values # state of california
X = inverted.drop(columns=3).values  # other states
print(X)
#%%
from sklearn.linear_model import Lasso
#help(Lasso)
weights_lr = Lasso(fit_intercept=False).fit(X, y).coef_
print(weights_lr.round(3))
#weights_lr.round(3) #round(3),保留小数点后三位

#%%
calif_synth_lr = (cigar.query("~california")
                  .pivot(index='year', columns="state")["cigsale"]
                  .values.dot(weights_lr))
print(calif_synth_lr)

#%%

plt.figure(figsize=(10,6))
plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"], label="California")
plt.plot(cigar.query("california")["year"], calif_synth_lr, label="Synthetic Control")
plt.vlines(x=1988, ymin=40, ymax=140, linestyle=":", lw=2, label="Proposition 99")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend();

#%%
plt.figure(figsize=(10,6))
plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth_lr,
         label="California Effect")
plt.vlines(x=1988, ymin=-30, ymax=7, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=2)
plt.title("State - Synthetic Across Time")
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.legend();
#%%
def synthetic_control(state, pool, data) -> np.array:
    features = ["cigsale", "retprice"]
    inverted = (data.query("~after_treatment") #~是对after_treatment 取反值，true就取为false
                .pivot(index='state', columns="year")[features]
                .T)
    y = inverted[state].values # treated
    X = inverted.drop(columns=state).values # donor pool
    weights = Lasso(fit_intercept=False).fit(X, y).coef_.round(8)
    synthetic = (data.query(f"~(state=={state})")
                 .pivot(index='year', columns="state")["cigsale"]
                 .values.dot(weights))
    return (data
            .query(f"state=={state}")[["state", "year", "cigsale", "after_treatment"]]
            .assign(synthetic=synthetic))

#%%
print(synthetic_control(1,1, cigar)) #这就是对state1进行合成控制法的命令

#%%
control_pool = cigar["state"].unique()
print(control_pool)

#%%
synthetic_states = [synthetic_control(state, control_pool, cigar) for state in control_pool]
print(synthetic_states[1])
#%%

plt.figure(figsize=(12,7))
for state in synthetic_states:
    plt.plot(state["year"], state["cigsale"] - state["synthetic"], color="C5",alpha=0.4)

plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth_lr,
        label="California");

plt.vlines(x=1988, ymin=-50, ymax=120, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=3)
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.title("State - Synthetic Across Time")
plt.legend();
#%%

def pre_treatment_error(state):
    pre_treat_error = (state.query("~after_treatment")["cigsale"] 
                       - state.query("~after_treatment")["synthetic"]) ** 2
    return pre_treat_error.mean()
#%%
plt.figure(figsize=(12,7))
for state in synthetic_states:
    
    # remove units with mean error above 5.
    if pre_treatment_error(state) < 5:
        plt.plot(state["year"], state["cigsale"] - state["synthetic"], color="C5",alpha=0.4)

plt.plot(cigar.query("california")["year"], cigar.query("california")["cigsale"] - calif_synth_lr,
        label="California");

plt.vlines(x=1988, ymin=-50, ymax=120, linestyle=":", lw=2, label="Proposition 99")
plt.hlines(y=0, xmin=1970, xmax=2000, lw=3)
plt.ylabel("Gap in per-capita cigarette sales (in packs)")
plt.title("Distribution of Effects")
plt.title("State - Synthetic Across Time (Large Pre-Treatment Errors Removed)")
plt.legend();
#%%
calif_number = 3

effects = [state.query("year==2000").iloc[0]["cigsale"] - state.query("year==2000").iloc[0]["synthetic"]
           for state in synthetic_states
           if pre_treatment_error(state) < 80] # filter out noise

calif_effect = cigar.query("california & year==2000").iloc[0]["cigsale"] - calif_synth_lr[-1] 

print("California Treatment Effect for the Year 2000:", calif_effect)
print(np.array(effects))
#%%

print(np.mean(np.array(effects) < calif_effect).raiseound(3))

#%%
_, bins, _ = plt.hist(effects, bins=10, color="C5", alpha=0.5);
plt.hist([calif_effect], bins=bins, color="C0", label="California")
plt.ylabel("Frquency")
plt.title("Distribution of Effects")
plt.legend();

