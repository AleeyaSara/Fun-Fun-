import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Streamlit App
st.title("Energy Efficiency Predictor for Household Appliances")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file):", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("## Dataset Preview")
    st.dataframe(data.head())

    # Display basic dataset statistics
    st.write("## Dataset Summary")
    st.write(data.describe())

    # Preprocessing
    st.write("## Data Preprocessing")
    # Encoding categorical data
    if 'Appliance Type' in data.columns:
        le = LabelEncoder()
        data['Appliance Type Encoded'] = le.fit_transform(data['Appliance Type'])
        st.write("Encoded Appliance Type:", le.classes_)

    # Normalize numerical data
    scaler = StandardScaler()
    data['Power Consumption Normalized'] = scaler.fit_transform(data[['Power Consumption']])

    # Preparing features and target
    features = ['Power Consumption Normalized', 'Appliance Type Encoded', 'Usage Pattern']
    target = 'Energy Efficiency Rating'

    X = data[features]
    y = data[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    st.write("## Model Training")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Model Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

    st.write("### Classification Report:")
    st.text(classification_report(y_test, predictions))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax)
    st.pyplot(fig)

    # Visualization
    st.write("## Data Visualization")
    st.write("### Energy Efficiency by Power Consumption")
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['Power Consumption'], data['Energy Efficiency Rating'], c='blue')
    ax.set_title('Energy Efficiency by Power Consumption')
    ax.set_xlabel('Power Consumption (kWh)')
    ax.set_ylabel('Energy Efficiency Rating')
    st.pyplot(fig)

    # Prediction Section
    st.write("## Make Predictions")
    power_consumption = st.number_input("Enter Power Consumption (kWh):", min_value=0.0, step=0.1)
    appliance_type = st.selectbox("Select Appliance Type:", le.classes_)
    usage_pattern = st.selectbox("Select Usage Pattern:", ["Occasional usage", "Daily usage", "Frequent usage"])

    if st.button("Predict Energy Efficiency Rating"):
        # Transform inputs
        appliance_encoded = le.transform([appliance_type])[0]
        usage_pattern_encoded = ["Occasional usage", "Daily usage", "Frequent usage"].index(usage_pattern)
        power_normalized = scaler.transform([[power_consumption]])[0][0]

        # Predict
        input_data = [[power_normalized, appliance_encoded, usage_pattern_encoded]]
        prediction = model.predict(input_data)[0]

        st.write(f"### Predicted Energy Efficiency Rating: {prediction}")













