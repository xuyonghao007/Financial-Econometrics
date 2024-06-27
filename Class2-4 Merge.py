# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:48:00 2022

@author: xuyonghao
"""

import os
import pandas as pd
import numpy as np
help( pd.DataFrame)
chang_path="D:/Python Basics/Class Data/"
#chang_path="D:\\Python Basics\\Class Data\\"
#chang_path="D:\\Python Basics\\Class Data"+"data.txt"


os.chdir(chang_path)
new_path=os.getcwd()

dict_data1={
    'location':  np.random.randint(1,20,size=200),  
    'date':  pd.date_range('20210101',periods=200),  #'用pd.date_range函数生成连续指定天数的的日期'
    'gender':np.random.randint(0,2,size=200),
    'grade': np.random.randint(1,5,size=200),
    'weight/kg':np.random.randint(40,50,size=200),
    'height/cm':np.random.randint(150,180,size=200)
    }
dataframe01 = pd.DataFrame(dict_data1)
#print(dataframe01)

file_name1=new_path+"\\Gender_data"+".csv"
dataframe01.to_csv(file_name1, index=False)
file_name2=new_path+"\\Gender_data"+".xlsx"
dataframe01.to_excel(file_name2, index=False)

dict_data2={
    'location':  range(11,31,1),  #range(start, stop[, step])
    'GDP':  np.random.randint(10,30,size=20),  #'用pd.date_range函数生成连续指定天数的的日期'
    #'gender':np.random.randint(0,2,size=20),
    }
dataframe02 = pd.DataFrame(dict_data2)
print(dataframe02)
#print(np.random.randint(1,500,5))

file_name2=new_path+"\\Location_data"+".csv"
dataframe02.to_csv(file_name2, index=False)


Data_gender=pd.read_csv(file_name1)
Data_loaction=pd.read_csv(file_name2)


whole_data2=pd.merge(left=Data_gender,right=Data_loaction, on="location" ,how="outer") #默认是inner
print(whole_data2)

file_name3=new_path+"\\Merge_data2"+".csv"
whole_data2.to_csv(file_name3, index=False)


