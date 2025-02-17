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

    # Check and transform dataset columns
    st.write("## Data Transformation")
    if 'Energy Consumption (kWh)' in data.columns:
        data['Power Consumption'] = data['Energy Consumption (kWh)']
    else:
        st.error("The dataset must include 'Energy Consumption (kWh)'.")

    if {'Television', 'Dryer', 'Oven', 'Refrigerator', 'Microwave'}.issubset(data.columns):
        data['Appliance Type'] = data[['Television', 'Dryer', 'Oven', 'Refrigerator', 'Microwave']].idxmax(axis=1)
    else:
        st.error("The dataset must include appliance columns: 'Television', 'Dryer', 'Oven', 'Refrigerator', 'Microwave'.")

    if 'Offloading Decision' in data.columns:
        data['Usage Pattern'] = data['Offloading Decision']
    else:
        st.error("The dataset must include 'Offloading Decision'.")

    # Assign energy efficiency ratings (example logic, adjust as needed)
    if 'Power Consumption' in data.columns:
        data['Energy Efficiency Rating'] = pd.cut(
            data['Power Consumption'],
            bins=[0, 1, 2, 3, float('inf')],
            labels=['A', 'B', 'C', 'D']
        )
    else:
        st.error("The dataset must include 'Power Consumption'.")

    # Ensure all required columns are present
    required_columns = ['Power Consumption', 'Appliance Type', 'Usage Pattern', 'Energy Efficiency Rating']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"The dataset is missing the following required columns: {', '.join(missing_columns)}")
    else:
        st.write("## Transformed Dataset")
        st.dataframe(data.head())

        # Preprocessing
        st.write("## Data Preprocessing")
        # Encoding categorical data
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

        # Check label distribution
        st.write("### Label Distribution in Training Set")
        st.write(y_train.value_counts())

        st.write("### Label Distribution in Test Set")
        st.write(y_test.value_counts())

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
        all_labels = ['A', 'B', 'C', 'D']  # Ensure all labels are included
        ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=all_labels, ax=ax)
        st.pyplot(fig)

        # Debugging Predictions
        st.write("### Predictions and True Labels")
        st.write(pd.DataFrame({'True Label': y_test, 'Prediction': predictions}).head())

        # Visualization
        st.write("## Data Visualization")
        st.write("### Energy Efficiency by Power Consumption")
        fig, ax = plt.subplots()
        if not pd.api.types.is_categorical_dtype(data['Energy Efficiency Rating']):
            data['Energy Efficiency Rating'] = pd.Categorical(data['Energy Efficiency Rating'], categories=['A', 'B', 'C', 'D'], ordered=True)
        scatter = ax.scatter(
            data['Power Consumption'],
            data['Energy Efficiency Rating'].cat.codes,
            c='blue'
        )
        ax.set_title('Energy Efficiency by Power Consumption')
        ax.set_xlabel('Power Consumption (kWh)')
        ax.set_ylabel('Energy Efficiency Rating')
        ax.set_yticks(range(len(data['Energy Efficiency Rating'].cat.categories)))
        ax.set_yticklabels(data['Energy Efficiency Rating'].cat.categories)
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
            input_data = pd.DataFrame([[power_normalized, appliance_encoded, usage_pattern_encoded]], columns=features)
            prediction = model.predict(input_data)[0]

            st.write(f"### Predicted Energy Efficiency Rating: {prediction}")















