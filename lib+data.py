### ----- ####
# LINK TO THE TUTORIAL:
# https://medium.com/analytics-vidhya/a-step-by-step-guide-to-generate-tabular-synthetic-dataset-with-gans-d55fc373c8db
### ----- ####

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import randn
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data = pd.read_csv('data/diabetes.csv')
data.shape      #(768, 9)
data.columns    # ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                # 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome')

 # MODEL SET UP
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
label = ['Outcome']
X = data[features]
y = data[label]

# RANDOM FOREST CLASSIFIER (The random forest classifier model is trained and evaluate the accuracy)
X_true_train, X_true_test, y_true_train, y_true_test = train_test_split(X, y, test_size=0.30, random_state=42)
clf_true = RandomForestClassifier(n_estimators=100)
clf_true.fit(X_true_train,y_true_train)
y_true_pred=clf_true.predict(X_true_test)

print("Base Accuracy:",metrics.accuracy_score(y_true_test, y_true_pred)) #0.7532467532467533
#The accuracy of the model trained from real data will be the base accuracy to compare
# with the model trained from generated fake data in the further steps.

print("Base classification report:",metrics.classification_report(y_true_test, y_true_pred))

#   Base classification report:               precision    recall  f1-score   support
#              0       0.81      0.81      0.81       151
#              1       0.64      0.65      0.65        80
#       accuracy                           0.75       231
#      macro avg       0.73      0.73      0.73       231
#   weighted avg       0.75      0.75      0.75       231

