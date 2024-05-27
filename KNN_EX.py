#importing libraries
import pandas as pd
from sklearn.metrics import accuracy_score

#importing dataset
dataset = pd.read_csv(r'C:\DATASCIENCE\MYPROJECCTS\Dhilsha_ML_works\Dataset\drug_classification.csv')
dataset.dropna(inplace=True)

#data preprocessing
print(dataset.isna().sum())

#Catgorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
dataset[['BP','Cholesterol','Na_to_K','Drug_Type']]=dataset[['BP','Cholesterol','Na_to_K','Drug_Type']].apply(LabelEncoder().fit_transform)
print(dataset)

#Extracting Independent and dependent Variable
x= dataset.iloc[:, [2,3,4]].values # selecting age and estimatedsalary
y= dataset.iloc[:, 5].values # purchase status
df2=pd.DataFrame(x)
print(df2.to_string())

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25,random_state=0)
print("x_train b4 scaling..")
df3=pd.DataFrame(x_train)
print(df3.to_string())

#feature scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
print("x_train after scaling...")
df4=pd.DataFrame(x_train)
print(df4.to_string())

#Fitting K-NN classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2 )
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred= classifier.predict(x_test)
print(y_pred)
print("Prediction comparison")
ddf=pd.DataFrame({"Y_test":y_test,"Y-pred":y_pred})
print(ddf.to_string())
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))