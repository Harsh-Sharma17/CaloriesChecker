import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

st.title("Calories and Weight Management Calculator")

# --- Load dataset and train/load model ---
model_file = "calorie_model.pkl"

if os.path.exists(model_file):
    # Load saved model
    model = joblib.load(model_file)
else:
    # Load dataset and train model
    df = pd.read_csv("calories.csv")

    # Encode gender
    df['Gender'] = df['Gender'].replace({'male': 0, 'female': 1})

    # Features and target
    X = df.drop("Calories", axis=1)
    y = df["Calories"]

    # Train linear regression model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model for future use
    joblib.dump(model, model_file)

# --- User Inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender_str = st.radio("Gender", ("Male", "Female"))
gender = 0 if gender_str == "Male" else 1
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
duration = st.number_input("Duration of exercise (minutes)", min_value=0, max_value=300, value=30)
heart_rate = st.number_input("Average Heart rate during exercise (bpm)", min_value=30, max_value=220, value=100)
body_temp = st.number_input("Body temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, format="%.1f")

# Prepare input dataframe
input_df = pd.DataFrame([[age, gender, height, weight, duration, heart_rate, body_temp]],
                        columns=['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

# --- Prediction & Calculations ---
if st.button("Calculate"):
    predicted_calories = model.predict(input_df)[0]
    predicted_calories = max(0, predicted_calories)  # Ensure calories aren't negative

    st.write(f"**Estimated calories burned during exercise:** {predicted_calories:.2f} kcal")

    # Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    if gender == 0:  # Male
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:  # Female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    st.write(f"**Estimated Basal Metabolic Rate (BMR):** {bmr:.2f} kcal/day")

    # Total Daily Energy Expenditure (TDEE)
    tdee = bmr + predicted_calories
    st.write(f"**Estimated Total Daily Energy Expenditure (TDEE):** {tdee:.2f} kcal/day")

    # Calories to maintain, lose, or gain weight
    st.write(f"**To maintain your weight:** {tdee:.2f} kcal/day")
    st.write(f"**To lose weight (~0.5 kg/week):** {tdee - 500:.2f} kcal/day")
    st.write(f"**To gain weight (~0.5 kg/week):** {tdee + 500:.2f} kcal/day")
