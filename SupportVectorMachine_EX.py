# importing libraries
import numpy as np
import pandas as pd

#importing dataset
data_set= pd.read_csv(r'C:\DATASCIENCE\MYPROJECCTS\Dhilsha_ML_works\Dataset\Tablet_press_data.csv')
df=pd.DataFrame(data_set)
print(df.to_string())

#checking the data for null value
print(data_set.isnull().sum())

#Extracting Independent and dependent Variable
x= data_set.iloc[:, [0,1,2,3,4,5,]].values
y= data_set.iloc[:, 6].values

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
from sklearn.svm import SVC # "Support vector classifier"
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred=classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("prediction status")
print(df2.to_string())

#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
