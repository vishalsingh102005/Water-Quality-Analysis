import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r"C:\Users\visha\OneDrive\Desktop\edunet project\Water Quality Testing.csv")

# Streamlit App Title
st.title("Water Quality Analysis App")

# Display Dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Feature Selection
st.sidebar.header("Select Features & Target")
all_columns = df.columns.tolist()

# Select Features and Target Column
X_features = st.sidebar.multiselect("Select Feature Columns (X):", all_columns, default=all_columns[:1])
y_target = st.sidebar.selectbox("Select Target Column (Y):", all_columns, index=all_columns.index("pH") if "pH" in all_columns else 0)

# Model Training
if st.button("Train Model"):
    if X_features and y_target:
        X = df[X_features]
        y = df[y_target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save model
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

        st.success("Model Trained Successfully!")
    else:
        st.error("Please select at least one feature and a target column.")

# Load Model for Prediction
if st.button("Predict Future pH Levels"):
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        # Generate future data dynamically with the same number of features
        future_data = np.array([[10] * len(X_features), [20] * len(X_features), [30] * len(X_features)])  # Example feature values
        predictions = model.predict(future_data)

        # Display Predictions
        results_df = pd.DataFrame(future_data, columns=X_features)  # Use selected feature names
        results_df["Predicted pH"] = predictions
        st.write(results_df)

    except Exception as e:
        st.error(f"Error: {e}. Please train the model first before making predictions.")
