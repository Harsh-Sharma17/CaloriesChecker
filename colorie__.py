import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset and train model (you can also load a saved model)
df = pd.read_csv(r"F:\HackNodeIndiaProject\CompleteCaloriesProject\calories.csv")

# Encode gender
df['Gender'] = df['Gender'].replace({'male': 0, 'female': 1})

# Prepare features and target
x = df.drop("Calories", axis=1)
y = df["Calories"]

# Train linear regression model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=50)
model = LinearRegression()
model.fit(x_train, y_train)

st.title("Calories and Weight Management Calculator")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.radio("Gender", (0, 1), format_func=lambda x: "Male" if x == 0 else "Female")
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
duration = st.number_input("Duration of exercise (minutes)", min_value=0, max_value=300, value=30)
heart_rate = st.number_input("Average Heart rate during exercise (bpm)", min_value=30, max_value=220, value=100)
body_temp = st.number_input("Body temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, format="%.1f")

# Prepare input for prediction
input_df = pd.DataFrame([[age, gender, height, weight, duration, heart_rate, body_temp]], columns=x.columns)

if st.button("Calculate"):
    predicted_calories = model.predict(input_df)[0]

    st.write(f"Estimated calories burned during exercise: {predicted_calories:.2f} kcal")

    # Basal Metabolic Rate (BMR) approximation using Mifflin-St Jeor Equation
    if gender == 0:  # Male
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:  # Female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    st.write(f"Estimated Basal Metabolic Rate (BMR): {bmr:.2f} kcal/day")

    # Total Daily Energy Expenditure (TDEE) estimation (using exercise duration roughly)
    tdee = bmr + predicted_calories
    st.write(f"Estimated Total Daily Energy Expenditure (TDEE): {tdee:.2f} kcal/day")

    # Calories to maintain weight (TDEE)
    st.write(f"To maintain your weight, consume approx: {tdee:.2f} kcal/day")

    # Calories to lose weight (500 kcal deficit for ~0.5 kg/week loss)
    cal_to_lose = tdee - 500
    st.write(f"To lose weight (~0.5 kg per week), consume approx: {cal_to_lose:.2f} kcal/day")

    # Calories to gain weight (500 kcal surplus for ~0.5 kg/week gain)
    cal_to_gain = tdee + 500
    st.write(f"To gain weight (~0.5 kg per week), consume approx: {cal_to_gain:.2f} kcal/day")
