import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.title('Energy Efficiency Predictor')

data = pd.read_csv('energy.csv')
st.write("Dataset Preview:", data.head())

# Preprocessing
data = data.dropna()  # Drop missing values
if 'Unix Timestamp' in data.columns:
    data = data.drop(columns=['Unix Timestamp'])

# Feature engineering
if 'Energy Consumption (kWh)' in data.columns and 'Apparent Power' in data.columns:
    data['Efficiency Index'] = data['Energy Consumption (kWh)'] / data['Apparent Power']

st.write("Processed Data:", data.head())

# Splitting the data
target_col = st.selectbox("Select the target column:", data.columns)
features = data.drop(columns=[target_col])
target = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Model evaluation
st.write("Accuracy:", accuracy_score(y_test, predictions))
st.write("Classification Report:", classification_report(y_test, predictions))

# Visualization
st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)
