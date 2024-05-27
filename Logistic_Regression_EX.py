# importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#importing datasets
data = pd.read_csv(r'C:\DATASCIENCE\MYPROJECCTS\Dhilsha_ML_works\Dataset\heart_failure_clinical_records_dataset.csv')
print(data.info())

#Data preprocessing
print(data.head())
print(data.isnull().sum())

#Extracting Independent and dependent Variable
x=data.iloc[:,0:10].values
y=data.iloc[:,11].values

# Splitting the dataset into training and test set.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
# fit the model with data
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
df2=pd.DataFrame(x_test)

#test data
print(df2.to_string())

#predicted data
print(y_pred)

#To compare the actual output values for X_test with the predicted value
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2.to_string())

#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


