import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Title and Description
st.title("Energy Efficiency Predictor")
st.write("This application predicts energy efficiency metrics using machine learning models.")

# File Uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Data Exploration
    st.write("### Data Exploration")
    st.write("Dataset Shape:", data.shape)
    st.write("Dataset Summary:")
    st.write(data.describe())

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Data Preprocessing
    st.write("### Data Preprocessing")
    target_column = st.selectbox("Select the target column:", data.columns)
    if target_column:
        features = data.drop(columns=[target_column])
        target = data[target_column]

        scaler_option = st.radio("Select a scaling method:", ("None", "Min-Max Scaler", "Standard Scaler"))

        if scaler_option == "Min-Max Scaler":
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
        elif scaler_option == "Standard Scaler":
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        st.write("Training set shape:", X_train.shape)
        st.write("Test set shape:", X_test.shape)

        # Model Training
        st.write("### Model Training")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model Evaluation
        st.write("### Model Evaluation")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)






