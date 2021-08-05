# IRIS-FLOWER-CLASSIFICATION---DECISION-TREE
#IMPORTING NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline 
#IMPORTING THE DATASET
df=pd.read_csv("/content/iris dataset.csv")
df.head()
#CHECKING THE DATA AND CLEANING
df.isnull().sum()
df=df.dropna()
df.isnull().sum()
df.shape
df.columns
df.info()
a=df.dtypes[df.dtypes=="object"]
a
df=df.drop_duplicates()
df.shape
#VISUALISING THE DATA
df.hist(figsize=(16,16),bins=10)
plt.show()
#SEPERATING X AND Y
x=df.iloc[:,df.columns!="class"]
y=df.iloc[:,df.columns=="class"]
x
y
#SPLITTING TEST AND TRAIN DATA
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
xtrain
ytrain
xtest
ytest
#DECISSION TREE ALGORITHM
from sklearn.tree import DecisionTreeClassifier
model1=DecisionTreeClassifier()
model1.fit(xtrain,ytrain)
model1_output=model1.predict(xtest)
model1_output
#CHECKING ACCURACY
from sklearn.metrics import accuracy_score
acc1=accuracy_score(model1_output,ytest)
acc1
#PREDICTING BY GIVING SAMPLE VALUES
predic1= model1.predict([[5.2,3.6,1.2,0.1]])
predic2=model1.predict([[6,2.3,4,1.2]])
predic3=model1.predict([[5.9,3,5.6,1.8]])
print(predic1)
print("\n")
print(predic2)
print("\n")
print(predic3)
