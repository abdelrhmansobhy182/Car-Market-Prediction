import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics


#read data
data = pd.read_csv('cars-test.csv')

#mean
data['volume(cm3)'] = data['volume(cm3)'].fillna(data['volume(cm3)'].mean())
data['volume(cm3)'] = data['volume(cm3)'].astype('int')

# get object
object = (data.dtypes == 'object')
object_cols = list(object[object].index);
# get missing data by mode
data[object_cols] = data[object_cols].fillna(data[object_cols].mode().iloc[0])
#data[object_cols] = data[object_cols].fillna(method ='pad')
new = data["car-info"].str.split(",", n=2, expand=True)
data["Model"] = new[0]
data["company"] = new[1]
data["date"] = new[2]
# Dropping old Name columns
data.drop(columns=["car-info"], inplace=True)
data['date'] = data['date'].str.replace('\W', '', regex=True)
data['date'] = pd.to_numeric(data['date'], errors='coerce')
data['Model'] = data['Model'].str.replace('\W', '', regex=True)
data['company'] = data['company'].str.replace('\W', '', regex=True)
print(data.isnull().sum())
# get object
object = (data.dtypes == 'object')
object_cols = list(object[object].index);
feature = ['date','segment', 'volume(cm3)', 'transmission' , 'Model']
# label encoder
pre = LabelEncoder()
for i in object_cols:
    data[i] = data[i].str.upper()
    data[i]= pre.fit_transform(data[i])
data.to_csv("test_all.csv", index=False)