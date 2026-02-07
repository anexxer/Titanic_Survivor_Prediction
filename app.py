import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('titanic_model.pkl')

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Input Fields
name = st.text_input("Passenger Name")
pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else f"{x}nd Class" if x==2 else f"{x}rd Class")
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("siblings / spouses aboard", 0, 10, 0)
parch = st.number_input("parents / children aboard", 0, 10, 0)

# Fare Dropdown (Simplified for UI)
# Using median fares for each class as representative values
fare_options = {
    "Cheap (3rd Class avg) - $8.05": 8.05,
    "Standard (2nd Class avg) - $14.25": 14.25,
    "Expensive (1st Class avg) - $60.30": 60.30,
    "Very Extravagant - $100+": 100.0
}
fare_label = st.selectbox("Fare Category", list(fare_options.keys()))
fare = fare_options[fare_label]

embarked_label = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
embarked = embarked_map[embarked_label]

# Preprocess Sex
sex_val = 0 if sex == "Male" else 1

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_val],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if name.lower() == "rose":
         st.success(f"Prediction: **Survived** (Confidence: JACK DIED FOR THIS)")
    elif name.lower() == "jack":
         st.error(f"Prediction: **Did Not Survive** (Confidence: Such a simp)")
    elif prediction == 1:
        st.success(f"Prediction: **Survived** (Confidence: {probability:.2%})")
    else:
        st.error(f"Prediction: **Did Not Survive** (Confidence: {probability:.2%})")
