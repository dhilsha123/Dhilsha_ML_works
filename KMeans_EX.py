import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r'C:\DATASCIENCE\MYPROJECCTS\Dhilsha_ML_works\Dataset\Mall_Customers.csv')
dataset.dropna(inplace=True)
print(dataset.isna().sum())



x= dataset.iloc[:, [2,3,4]].values # selecting age and estimatedsalary

from sklearn.cluster import KMeans
wcss= [] #Initializing the list for the values of WCSS
#Using for loop for iterations from 1 to 10.
for i in range(1, 9):
  kmeans = KMeans(n_clusters= i, init='k-means++', random_state= 42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(x)
print(y_predict)

mtp.plot(range(1, 9), wcss)
mtp.title('The Elobw Method Graph')
mtp.xlabel('Number of clusters(k)')
mtp.ylabel('wcss_list')
mtp.show()
#training the
