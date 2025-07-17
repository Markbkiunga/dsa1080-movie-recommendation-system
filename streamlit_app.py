import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Traffic Accident Severity Predictor", layout="centered")

# --- Load the trained model and feature columns ---
MODEL_PATH = os.path.join('models', 'random_forest_model.pkl')
FEATURES_PATH = os.path.join('models', 'feature_columns.pkl')

@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

model, feature_columns = load_model_and_features()

# --- Title and Instructions ---
st.title("🚦 Traffic Accident Severity Predictor")
st.markdown("""
Use this app to predict the **severity of a traffic accident** based on key factors.  
Fill in the details below and click **Predict** to see the result.
""")

# --- Input UI ---
st.subheader("📋 Accident Details")

# Example features: replace/add more based on your dataset
age_band = st.selectbox("Age Band of Driver", [
    "Under 18", "18-30", "31-50", "Over 51", "Unknown"
])

sex = st.selectbox("Sex of Driver", ["Male", "Female", "Unknown"])

education = st.selectbox("Educational Level", [
    "Below High School", "High School", "Above High School", "Unknown"
])

driving_experience = st.selectbox("Driving Experience", [
    "No Experience", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "Unknown"
])

weather = st.selectbox("Weather Conditions", [
    "Clear", "Rainy", "Cloudy", "Fog", "Windy", "Other"
])

light = st.selectbox("Light Conditions", [
    "Daylight", "Darkness - lights lit", "Darkness - lights unlit",
    "Darkness - no lighting", "Unknown"
])

collision = st.selectbox("Type of Collision", [
    "Vehicle with vehicle", "Vehicle with pedestrian", "Vehicle with object",
    "Rollover", "Fall from vehicle", "Other"
])

cause = st.selectbox("Cause of Accident", [
    "Overspeed", "Overtaking", "Driving under influence", "Fatigue", 
    "Vehicle defect", "Distracted driving", "Unknown", "Other"
])

# --- Convert inputs to a feature vector ---
input_dict = {col: 0 for col in feature_columns}

# Dynamically set one-hot encoded features to 1
def set_feature(prefix, value):
    col_name = f"{prefix}_{value}"
    if col_name in input_dict:
        input_dict[col_name] = 1

set_feature("Age_band_of_driver", age_band)
set_feature("Sex_of_driver", sex)
set_feature("Educational_level", education)
set_feature("Driving_experience", driving_experience)
set_feature("Weather_conditions", weather)
set_feature("Light_conditions", light)
set_feature("Type_of_collision", collision)
set_feature("Cause_of_accident", cause)

# --- Convert to DataFrame ---
input_df = pd.DataFrame([input_dict])


# --- Predict ---
if st.button("🔍 Predict Severity"):
    prediction = model.predict(input_df)[0]
    labels = {
        0: "Slight Injury",
        1: "Serious Injury",
        2: "Fatal Injury"
    }
    st.success(f"🚨 **Predicted Severity:** {labels[prediction]}")