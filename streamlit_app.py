import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Streamlit app title
st.title("Energy Efficiency Predictor")

# Sidebar for user input
st.sidebar.header("User Input")

# File uploader for the dataset
data_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if data_file:
    data = pd.read_csv(data_file)

    # Display dataset preview
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Display dataset info
    st.subheader("Dataset Information")
    buffer = []
    data.info(buf=buffer)
    st.text("\n".join(buffer))

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    # Data visualization
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Feature selection and preprocessing
    target_column = st.sidebar.selectbox("Select Target Column", options=data.columns)
    feature_columns = st.sidebar.multiselect("Select Feature Columns", options=[col for col in data.columns if col != target_column])

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Scaling options
        scaling_method = st.sidebar.selectbox("Select Scaling Method", ["None", "MinMaxScaler", "StandardScaler"])
        if scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif scaling_method == "StandardScaler":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Train-test split
        test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model training
        st.subheader("Model Training")
        n_estimators = st.sidebar.slider("Number of Trees in Random Forest", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

else:
    st.write("Please upload a dataset to get started.")







