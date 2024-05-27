
# importing libraries
import pandas as pd
import numpy as np


#importing datasets
data_set = pd.read_csv(r'C:\DATASCIENCE\MYPROJECCTS\Dhilsha_ML_works\Dataset\smartphones - smartphones.csv')
print(data_set.to_string())

#Data preprocessing
data_set.info()
print(data_set.isnull().sum())
print(data_set.dropna(inplace=True))
print(data_set.isnull().sum())

#Catgorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()

data_set[['model','price','sim','processor','ram','battery','display','camera','card','os']]=data_set[['model','price','sim','processor','ram','battery','display','camera','card','os']].apply(LabelEncoder().fit_transform)
print(data_set)

#Extracting Independent and dependent Variable
x = data_set.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9,10]].values
y = data_set.iloc[:, 1].values

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result;
y_pred = regressor.predict(x_test)

#To compare the actual output values for X_test with the predicted value
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df.to_string())

print("Mean")
print(data_set.describe())

#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# predicting the accuracy score
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")
k_test=[[1.0,0.0,142107.34,91391.77,366168.42]]




