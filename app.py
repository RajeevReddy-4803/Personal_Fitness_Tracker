import streamlit as st
import numpy as np
import pandas as pd
import time
import joblib  # To load trained model

import warnings
warnings.filterwarnings('ignore')

# Set page config with a sleek theme
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #ff4b4b;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px;
    }
    .stSlider>div>div>div {
        background-color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------- HEADER -----------------------------
st.title("🏋️‍♂️ Personal Fitness Tracker 🔥")
st.markdown(
    "Track your fitness journey! Predict **calories burned** based on your personal stats like `Age`, `Gender`, `BMI`, and more."
)

# Sidebar with enhanced styling
st.sidebar.markdown("### User Input Parameters 🔍", unsafe_allow_html=True)
st.sidebar.markdown("Adjust the sliders to match your stats!")

# ----------------------------- USER INPUT FUNCTION -----------------------------
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30, help="Your age in years")
    bmi = st.sidebar.slider("BMI", 15, 40, 20, help="Body Mass Index")
    duration = st.sidebar.slider("Duration (min)", 0, 60, 30, help="Exercise duration in minutes")
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 200, 90, help="Average heart rate during exercise")
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 37.5, help="Your body temperature")
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"), help="Select your gender")

    gender = 1 if gender_button == "Male" else 0

    user_data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Encoding gender as 1 for Male, 0 for Female
    }

    return pd.DataFrame(user_data, index=[0])

df = user_input_features()

# ----------------------------- DISPLAY USER INPUT -----------------------------
st.markdown("---")
st.header("📝 Your Parameters")
with st.spinner("Loading your stats..."):
    time.sleep(1)
    st.dataframe(df.style.set_properties(**{'background-color': '#ffe6e6', 'border-color': '#ff4b4b', 'color': '#333'}))

# ----------------------------- LOAD TRAINED MODEL -----------------------------
@st.cache_resource
def load_model():
    try:
        model_path = "trained_model.pkl"  # Ensure this file exists
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file 'trained_model.pkl' not found. Please upload the model file.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------- MAKE PREDICTION -----------------------------
st.markdown("---")
st.header("🔥 Predicted Calories Burned")
if model is not None:
    with st.spinner("Calculating calories..."):
        # Default feature names if model.feature_names_in_ is unavailable
        expected_features = ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_male"]
        df = df.reindex(columns=expected_features, fill_value=0)
        try:
            prediction = model.predict(df)
            time.sleep(1)
            st.markdown(f"### 🎉 **{round(prediction[0], 2)} kilocalories** 🔥", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prediction = None
else:
    st.warning("Prediction unavailable due to missing model.")
    prediction = None

# ----------------------------- LOAD DATASETS -----------------------------
st.markdown("---")
st.header("🔄 Similar Cases")
with st.expander("View Similar Fitness Profiles"):
    try:
        calories = pd.read_csv("calories.csv")
        exercise = pd.read_csv("exercise.csv")

        # Merge datasets for full fitness data
        exercise_df = exercise.merge(calories, on="User_ID")
        exercise_df.drop(columns="User_ID", inplace=True)

        # Add BMI column to dataset
        exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

        # Find similar cases based on predicted calories
        if prediction is not None:
            calorie_range = [prediction[0] - 10, prediction[0] + 10]
            similar_data = exercise_df[
                (exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])
            ]
            st.markdown("#### 🔍 Showing 5 users with similar calorie burns:")
            if not similar_data.empty:
                st.dataframe(similar_data.sample(min(5, len(similar_data))).style.set_properties(**{'background-color': '#fff0f0', 'border-color': '#ff4b4b'}))
            else:
                st.write("No similar cases found within the calorie range.")
        else:
            st.write("Cannot find similar cases without a valid prediction.")
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}. Please ensure 'calories.csv' and 'exercise.csv' are available.")
    except Exception as e:
        st.error(f"Error processing datasets: {e}")

# ----------------------------- GENERAL INFORMATION -----------------------------
st.markdown("---")
st.header("📊 Insights Compared to Others")
try:
    col1, col2 = st.columns(2)
    with col1:
        bool_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
        bool_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
        st.markdown(f"🟢 You are older than **{round(sum(bool_age) / len(bool_age) * 100, 2)}%** of users.")
        st.markdown(f"🟢 Your exercise duration is longer than **{round(sum(bool_duration) / len(bool_duration) * 100, 2)}%** of users.")

    with col2:
        bool_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
        bool_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()
        st.markdown(f"🟢 Your heart rate is higher than **{round(sum(bool_heart_rate) / len(bool_heart_rate) * 100, 2)}%** of users.")
        st.markdown(f"🟢 Your body temp is higher than **{round(sum(bool_body_temp) / len(bool_body_temp) * 100, 2)}%** of users.")
except NameError:
    st.warning("Insights unavailable due to missing dataset.")
except Exception as e:
    st.error(f"Error generating insights: {e}")

# ----------------------------- FOOTER -----------------------------
st.markdown("---")
st.markdown("### 🚀 Stay Fit & Track Your Progress Daily! 💪", unsafe_allow_html=True)
st.balloons()  # Fun animation