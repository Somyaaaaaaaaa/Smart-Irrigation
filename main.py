import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("data.csv")

print("dataset preview:")
print(df.head()) #prints the first 5 rows

print(df.info()) #the datatype of the data 

print(df.columns) #shows the columns present in the dataset

df= df.drop('Unnamed: 0', axis=1) #deletes the column
print(df.head())

print(df.describe().T) #shows how the data is distributed
#T transposes the data so the rowas turn to columns

#feature and target selection
#iloc refers to integer location
X= df.iloc[:, 0:20] #features, independant variable 
y= df.iloc[:, 20:] #labels/target, dependant variable

#for all the rows, we write [:]

print(X.sample(10))

print(y.sample(10))

print(X.info())

print(y.info())

print(X.shape, y.shape)