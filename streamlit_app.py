import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Energy Efficiency Predictor')

data = pd.read_csv('energy.csv')

st.write("Dataset Overview:")
st.write(data.describe())
st.write(data.info())

data = data.dropna()
if 'Unix Timestamp' in data.columns:
    data = data.drop(columns=['Unix Timestamp'])

data['Efficiency Index'] = data['Energy Consumption (kWh)'] / data['Apparent Power']

st.write("Processed Data:", data.head())

# Simple train-test split
target_col = st.selectbox("Select the target column:", data.columns)
features = data.drop(columns=[target_col])
target = data[target_col]

train_size = int(0.8 * len(data))
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

# Simple threshold-based model as a placeholder
threshold = y_train.mean()
predictions = [1 if x > threshold else 0 for x in y_test]

# Basic accuracy metric
accuracy = sum([1 for pred, actual in zip(predictions, y_test) if pred == actual]) / len(y_test)
st.write("Accuracy:", accuracy)

# Confusion matrix
cm = pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(predictions, name='Predicted'))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

