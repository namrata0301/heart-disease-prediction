#importing libraries
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# loading the dataset to a pandas DataFrame
df = pd.read_csv('heart_dataset.csv')
data = df.copy()

# print the first 5 rows of the dataframe
data.head()

# number of rows and columns in the dataset
data.shape

# getting some info about the data
data.info()

# counting the number of missing values in the dataset
data.isnull().sum()

# statistical measures about the data
data.describe()


#get correlations of each features in dataset
x = data.corr()
plt.figure(figsize=(20,20))
#plot heat map
sns.heatmap(x ,annot=True,cmap="RdYlGn")

sns.set_style('whitegrid')
sns.countplot(x='target',data=data,palette='RdBu_r')


# checking the distribution of Target Variable
data['target'].value_counts()

# separating the data & target
X = data.drop(columns='target',axis=1)
Y = data['target']

print(X)
print(Y)


#Splitting the Data into Training data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#performing Standardization
scaler= StandardScaler()
X_train_scaler= scaler.fit_transform(X_train)
X_test_scaler= scaler.fit_transform(X_test)


#Model Training
#RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
model.fit(X_train_scaler, Y_train)


#Model Evaluation
# accuracy on test data
Y_pred= model.predict(X_test_scaler)
print('Accuracy: ', accuracy_score(Y_test, Y_pred))


#creating confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

filename = 'heart-disease-prediction-model.pkl'
pickle.dump(model, open(filename, 'wb'))
