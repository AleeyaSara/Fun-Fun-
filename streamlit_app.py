import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import io

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
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    # Data visualization
    st.subheader("Feature Relationships")

    # Scatter plot example: Power Consumption vs Energy Efficiency Rating
    if "Power Consumption" in data.columns and "Energy Efficiency Rating" in data.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(x="Power Consumption", y="Energy Efficiency Rating", data=data, ax=ax)
        ax.set_title("Power Consumption vs Energy Efficiency Rating")
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    selected_features = st.sidebar.multiselect("Select Features for Heatmap", options=data.columns, default=data.columns[:5])

    if selected_features:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = data[selected_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, annot_kws={"size": 8})
        ax.set_title("Correlation Heatmap", fontsize=14)
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8)
        st.pyplot(fig)

    # Feature selection and preprocessing
    target_column = st.sidebar.selectbox("Select Target Column", options=data.columns)
    feature_columns = st.sidebar.multiselect("Select Feature Columns", options=[col for col in data.columns if col != target_column])

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Encode categorical target if necessary
        if y.dtypes == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

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

        # Prediction tool
        st.subheader("Predict Energy Efficiency Rating")
        input_data = {}
        for col in feature_columns:
            if col in data.select_dtypes(include=['object']).columns:
                options = data[col].unique()
                input_data[col] = st.selectbox(f"Select {col}", options=options)
            else:
                input_data[col] = st.number_input(f"Enter {col}", value=0.0)

        input_df = pd.DataFrame([input_data])

        # Encode and scale input data
        for col in input_df.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            input_df[col] = encoder.fit_transform(input_df[col])

        if scaling_method != "None":
            input_df = scaler.transform(input_df)

        prediction = model.predict(input_df)
        if y.dtypes == 'object':
            prediction = label_encoder.inverse_transform(prediction)

        st.write(f"Predicted Energy Efficiency Rating: {prediction[0]}")

else:
    st.write("Please upload a dataset to get started.")












