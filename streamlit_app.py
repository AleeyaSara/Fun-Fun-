import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import traceback

# Streamlit App
st.title("Energy Efficiency Predictor")

# Direct file path for testing
uploaded_file = "energy.csv"  # Replace file uploader for testing

try:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(data.head())

    # Display basic dataset info
    st.write("### Dataset Description:")
    st.write(data.describe())

    # Handle missing values
    if data.isnull().values.any():
        st.write("### Handling Missing Values:")
        data = data.fillna(data.median())
        st.write("Missing values filled with column median.")

    # Data preprocessing
    st.sidebar.header("Preprocessing Options")
    scaler_choice = st.sidebar.selectbox("Select Scaler:", ["None", "MinMaxScaler", "StandardScaler"])

    if scaler_choice != "None":
        st.write(f"### Applying {scaler_choice} to the dataset")
        scaler = MinMaxScaler() if scaler_choice == "MinMaxScaler" else StandardScaler()
        data_scaled = scaler.fit_transform(data.select_dtypes(include=np.number))
        data = pd.DataFrame(data_scaled, columns=data.select_dtypes(include=np.number).columns)
        st.write("### Scaled Dataset Preview:")
        st.write(data.head())

    # Sidebar for target selection
    target_column = st.sidebar.selectbox("Select Target Column:", data.columns)

    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        st.sidebar.header("Model Training")
        n_estimators = st.sidebar.slider("Number of Trees in Random Forest", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.2f}")
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        st.write("### Confusion Matrix:")
        confusion = confusion_matrix(y_test, y_pred)
        st.write(confusion)

        # Feature importance
        st.write("### Feature Importances:")
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(feature_importances)

except Exception as e:
    st.error("An error occurred while processing the file:")
    st.text(traceback.format_exc())





