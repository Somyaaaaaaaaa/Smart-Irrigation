import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

data = pd.read_csv("data.csv")

print("Dataset Preview:")
print(data.head(), "\n")

print("Dataset Info:")
print(data.info(), "\n")

print("Columns:")
print(data.columns, "\n")

if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

print("After Cleaning:")
print(data.head(), "\n")

print("Summary Statistics:")
print(data.describe(), "\n")

features = data[[col for col in data.columns if 'sensor' in col]]
labels = data[[col for col in data.columns if 'parcel' in col]]

print("Sample Features:")
print(features.sample(10, random_state=42), "\n")

print("Sample Labels:")
print(labels.sample(10, random_state=42), "\n")

print("Feature Info:")
print(features.info(), "\n")

print("Label Info:")
print(labels.info(), "\n")

print("Shapes:", features.shape, labels.shape, "\n")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=labels.columns))

print("Parcel Activation Summary:")
print(labels.sum())

joblib.dump(model, "Farm_Irrigation_System.pkl")
print("Model saved as Farm_Irrigation_System.pkl")
