# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import numpy as np

#importing datasets
data_set= pd.read_csv(r'C:\DATASCIENCE\MYPROJECCTS\Dhilsha_ML_works\Dataset\winequality-red.csv')
df=pd.DataFrame(data_set)
print("Actual Dataset")
print(df.to_string())


x = data_set.iloc[:, 0:11].values
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='ward')
y_pred= hc.fit_predict(x)
print(y_pred)




