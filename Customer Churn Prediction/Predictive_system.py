#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:39:16 2025

@author: divya
"""

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
with open('/Users/divya/Desktop/Main ML Projects/CustomerChurnPrediction(Bank)/Streamlit/trained_pipeline.sav', 'rb') as file:
    loaded_model = pickle.load(file)


# Example of new data (ensure it's processed like the training data)
new_data = pd.DataFrame({
    'CreditScore': [250],
    'Geography': ['Germany'],
    'Gender': ['Female'],
    'Age': [70],
    'Tenure': [1],
    'Balance': [0],
    'NumOfProducts': [1],
    'HasCrCard': [0],
    'IsActiveMember': [0],
    'EstimatedSalary': [0]
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