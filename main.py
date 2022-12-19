import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


from Pre_processing import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics


#read data
data = pd.read_csv('cars-train.csv')
print(data.isnull().sum())
#mean
data['volume(cm3)'] = data['volume(cm3)'].fillna(data['volume(cm3)'].mean())
data['volume(cm3)'] = data['volume(cm3)'].astype('int')

# get object
object = (data.dtypes == 'object')
object_cols = list(object[object].index);
# get missing data by mode
data[object_cols] = data[object_cols].fillna(data[object_cols].mode().iloc[0])
#data['drive_unit']=data['drive_unit'].interpolate(method ='linear', limit_direction ='forward')
#data.dropna(how='any',inplace=True)

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
data = data.reindex(columns = [col for col in data.columns if col != 'price(USD)'] + ['price(USD)'])

# get object
object = (data.dtypes == 'object')
object_cols = list(object[object].index);
feature = ['date','segment', 'volume(cm3)', 'transmission' ]
# label encoder,

pre = LabelEncoder()
for i in object_cols:
    data[i]= pre.fit_transform(data[i])



data.to_csv("all.csv", index=False)

#############################

X=data[feature] #Features

Y=data['price(USD)'] #Label
X,minimum,maximum = featureScaling(X,0,1)

corr = data.corr()

#Correlation plot
#plt.subplots(figsize=(12, 8))
#top_corr =corr
#sns.heatmap(top_corr, annot=True)
#plt.show()
#Apply Linear Regression on the selected features
cls = linear_model.LinearRegression()
cls.fit(X,Y)
Linear_prediction= cls.predict(X)
print("Model Accuracy(%): \t" + str(r2_score(Y, Linear_prediction)*100) + "%")
print('Mean Square Error Linear', metrics.mean_squared_error(np.asarray(Y), Linear_prediction))


#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15,shuffle=True,random_state=10)
poly_features = PolynomialFeatures(degree=6)
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict( poly_features.fit_transform(X_train))
prediction_2 = poly_model.predict( poly_features.fit_transform(X_test))


print('Mean Square Error', metrics.mean_squared_error(y_train, prediction))
print("Model Accuracy(%): \t" + str(r2_score(y_train, prediction)*100) + "%")
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction_2))
print(Y[5])
print(prediction[5])
#read
test_data=pd.read_csv('test_all.csv')
x_pred=test_data[feature]
x_pred=featureScaling_test(x_pred,0,1,minimum,maximum)
prediction_2 = poly_model.predict( poly_features.fit_transform(x_pred))
submission=pd.DataFrame()
#submission["car_id"]=data_cleaned["car_id"]
submission["price(USD)"]=prediction_2
submission.to_csv('submission.csv')
