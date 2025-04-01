# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 17:49:38 2025

@author: Divya
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import pickle


# Load the saved model
with open('C:/Users/Divya/OneDrive/Pictures/Desktop/Streamlit/trained_pipeline.sav', 'rb') as file:
    loaded_model = pickle.load(file)


# Example of new data (ensure it's processed like the training data)
new_data = pd.DataFrame({
    'CreditScore': [700],
    'Geography': ['Germany'],
    'Gender': ['Female'],
    'Age': [35],
    'Tenure': [5],
    'Balance': [75000],
    'NumOfProducts': [1],
    'HasCrCard': [1],
    'IsActiveMember': [0],
    'EstimatedSalary': [90000]
})

# Make predictions using the loaded model
prediction = loaded_model.predict(new_data)

# Output the prediction
print("Predicted Class:", prediction)

# Output result
if prediction[0] == 1:
    print("Exited")
else:
    print("Did Not Exit")