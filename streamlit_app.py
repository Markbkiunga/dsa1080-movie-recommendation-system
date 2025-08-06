import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load('./models/random_forest_model.pkl')
feature_columns = joblib.load('./models/feature_columns.pkl')

# Define input form
st.title("🚦 Traffic Accident Severity Prediction")
st.write("Fill in the information below to predict accident severity:")

with st.form("severity_form"):
    age = st.selectbox("Age Band of Driver", ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown'])
    sex = st.selectbox("Sex of Driver", ['Male', 'Female', 'Unknown'])
    education = st.selectbox("Educational Level", ['Below High School', 'High School', 'Above high school', 'Unknown'])
    experience = st.selectbox("Driving Experience", ['No Experience', '1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'Unknown'])
    lane = st.selectbox("Lanes or Medians", ['Two-way (divided with broken lines)', 'Undivided Two way', 'One way', 'Unknown'])
    junction = st.selectbox("Types of Junction", ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'T Shape', 'Unknown'])
    road = st.selectbox("Road Surface Type", ['Asphalt', 'Gravel', 'Earth', 'Other'])
    light = st.selectbox("Light Conditions", ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit'])
    weather = st.selectbox("Weather Conditions", ['Clear', 'Raining', 'Fog', 'Windy', 'Other'])
    collision = st.selectbox("Type of Collision", ['Vehicle with vehicle front impact', 'With pedestrian', 'With Train', 'Vehicle with vehicle rear-end collision'])
    movement = st.selectbox("Vehicle Movement", ['Moving Forward', 'Reversing', 'Stopping', 'Parked'])
    pedestrian = st.selectbox("Pedestrian Movement", ['Not a Pedestrian', 'Crossing from right', 'Crossing from left'])
    cause = st.selectbox("Cause of Accident", ['No distancing', 'Overspeed', 'Changing lane', 'Other'])
    relation = st.selectbox("Vehicle Driver Relation", ['Owner of vehicle', 'Employee', 'Unknown'])

    submit = st.form_submit_button("Predict")

if submit:
    # Step 1: Create DataFrame from input
    input_dict = {
        'Age_band_of_driver': age,
        'Sex_of_driver': sex,
        'Educational_level': education,
        'Driving_experience': experience,
        'Lanes_or_Medians': lane,
        'Types_of_Junction': junction,
        'Road_surface_type': road,
        'Light_conditions': light,
        'Weather_conditions': weather,
        'Type_of_collision': collision,
        'Vehicle_movement': movement,
        'Pedestrian_movement': pedestrian,
        'Cause_of_accident': cause,
        'Vehicle_driver_relation': relation
    }

    input_df = pd.DataFrame([input_dict])

    # Step 2: One-hot encode using same structure as training data
    encoded_input = pd.get_dummies(input_df)
    encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)

    # Step 3: Predict
    prediction = model.predict(encoded_input)[0]
    probs = model.predict_proba(encoded_input)[0]

    # Step 4: Show results
    severity_map = {0: "Slight Injury", 1: "Serious Injury", 2: "Fatal Injury"}
    st.subheader(f"🚑 Predicted Severity: {severity_map[prediction]}")
    st.write("📊 Prediction Probabilities:")
    st.json({
        "Slight Injury": f"{probs[0]:.2%}",
        "Serious Injury": f"{probs[1]:.2%}",
        "Fatal Injury": f"{probs[2]:.2%}"
    })