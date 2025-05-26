import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    :root {
        --primary: #4CAF50;
        --primary-light: #81C784;
        --secondary: #2196F3;
        --accent: #FF9800;
        --background: #1E1E1E;
        --card-bg: #2D2D2D;
        --text: #FFFFFF;
        --text-secondary: #CCCCCC;
    }
    .main { background-color: var(--background); color: var(--text); }
    .sidebar .sidebar-content { background-color: var(--card-bg); padding: 1.5rem; }
    h1, h2, h3 { color: var(--primary); font-family: 'Arial', sans-serif; font-weight: 600; }
    .stDataFrame, .stMetric { 
        background-color: var(--card-bg); 
        border-radius: 10px; 
        padding: 1rem;
        border: 1px solid var(--primary);
        margin: 1rem 0;
    }
    .stButton>button { 
        background-color: var(--primary); 
        color: white; 
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: var(--primary-light);
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--primary);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--accent);
    }
    .section-spacer {
        margin: 2rem 0;
        padding: 1rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Sample dataset for fallback
SAMPLE_DATA = pd.DataFrame({
    'User_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [25, 30, 35, 28, 32, 27, 31, 29, 33, 26],
    'Gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'Height': [175, 165, 180, 170, 178, 168, 182, 172, 176, 169],
    'Weight': [70, 60, 85, 65, 75, 62, 80, 68, 72, 64],
    'Duration': [30, 45, 20, 35, 40, 25, 50, 30, 45, 35],
    'Heart_Rate': [120, 130, 110, 125, 135, 115, 140, 120, 130, 125],
    'Body_Temp': [37.2, 37.5, 37.0, 37.3, 37.6, 37.1, 37.7, 37.2, 37.4, 37.3],
    'Calories': [250, 300, 200, 275, 325, 225, 350, 250, 300, 275]
})

# Header with enhanced caption
st.title("🏋️‍♂️ Personal Fitness Tracker 🔥")
st.markdown("""
    <div style='text-align: center; margin: 2rem 0; padding: 1.5rem; background-color: var(--card-bg); border-radius: 10px; border: 1px solid var(--primary);'>
        <h2 style='color: var(--primary); margin-bottom: 1rem;'>Your Personal Fitness Journey Starts Here</h2>
        <p style='font-size: 1.2rem; color: var(--text-secondary); line-height: 1.6;'>
            Transform your workouts with AI-powered insights. Track calories, monitor progress, and achieve your fitness goals with precision.
        </p>
        <div style='margin-top: 1rem; display: flex; justify-content: center; gap: 1rem;'>
            <span style='color: var(--accent);'>🎯 Smart Predictions</span>
            <span style='color: var(--secondary);'>📊 Detailed Analytics</span>
            <span style='color: var(--primary-light);'>💪 Personalized Insights</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Initialize session states
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'user_history' not in st.session_state:
    st.session_state.user_history = []
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {
        'dates': [], 'calories': [], 'bmi': [],
        'duration': [], 'heart_rate': [], 'body_temp': []
    }

# Data persistence functions
def save_to_json(data, filename='fitness_history.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"Error saving data: {e}")

def load_from_json(filename='fitness_history.json'):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load historical data
if not st.session_state.user_history:
    historical_data = load_from_json()
    if historical_data:
        st.session_state.user_history = historical_data

# User input features with enhanced UI
st.sidebar.header("User Input Parameters 🔍")
st.sidebar.markdown("Adjust the sliders to match your stats!")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30, help="Your age in years")
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 20.0, help="Body Mass Index")
    duration = st.sidebar.slider("Duration (min)", 0, 60, 30, help="Exercise duration in minutes")
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 180, 90, help="Average heart rate during exercise")
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 40.0, 37.5, help="Your body temperature")
    gender = 1 if st.sidebar.radio("Gender", ("Male", "Female")) == "Male" else 0
    
    # Enhanced validation with detailed feedback
    validation_messages = []
    if bmi < 18.5:
        validation_messages.append("⚠️ BMI is below healthy range (18.5)")
    elif bmi > 25:
        validation_messages.append("⚠️ BMI is above healthy range (25)")
    if heart_rate > 160:
        validation_messages.append("⚠️ Heart rate is unusually high (>160 bpm)")
    if body_temp > 39.0:
        validation_messages.append("⚠️ Body temperature is unusually high (>39°C)")
    
    for msg in validation_messages:
        st.sidebar.warning(msg)
    
    return pd.DataFrame({
        "Age": age, "BMI": bmi, "Duration": duration,
        "Heart_Rate": heart_rate, "Body_Temp": body_temp,
        "Gender_male": gender
    }, index=[0])

df = user_input_features()

# Display user input with enhanced metrics
st.header("📝 Your Parameters")
with st.spinner("Loading your stats..."):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h3>Basic Information</h3>
                <p><strong>Age:</strong> {} years</p>
                <p><strong>BMI:</strong> {}</p>
                <p><strong>Gender:</strong> {}</p>
            </div>
        """.format(
            df['Age'].values[0],
            df['BMI'].values[0],
            'Male' if df['Gender_male'].values[0] == 1 else 'Female'
        ), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h3>Exercise Metrics</h3>
                <p><strong>Duration:</strong> {} minutes</p>
                <p><strong>Heart Rate:</strong> {} bpm</p>
                <p><strong>Body Temperature:</strong> {}°C</p>
            </div>
        """.format(
            df['Duration'].values[0],
            df['Heart_Rate'].values[0],
            df['Body_Temp'].values[0]
        ), unsafe_allow_html=True)
    with col3:
        # Calculate and display intensity level
        intensity = "Low" if df['Heart_Rate'].values[0] < 100 else "Medium" if df['Heart_Rate'].values[0] < 140 else "High"
        st.markdown("""
            <div class='metric-card'>
                <h3>Workout Intensity</h3>
                <p><strong>Level:</strong> {}</p>
                <p><strong>Target Zone:</strong> {}%</p>
                <p><strong>Calorie Impact:</strong> {}</p>
            </div>
        """.format(
            intensity,
            "60-70" if intensity == "Low" else "70-85" if intensity == "Medium" else "85-100",
            "Low" if intensity == "Low" else "Medium" if intensity == "Medium" else "High"
        ), unsafe_allow_html=True)

# Load model
@st.cache_resource(ttl=3600)
def load_model():
    try:
        return joblib.load("trained_model.pkl")
    except FileNotFoundError:
        st.error("Model file 'trained_model.pkl' not found.")
        return None

model = load_model()

# Load datasets
@st.cache_data(ttl=3600)
def load_datasets():
    try:
        calories = pd.read_csv("calories.csv")
        exercise = pd.read_csv("exercise.csv")
        exercise_df = exercise.merge(calories, on="User_ID")
        exercise_df.drop(columns="User_ID", inplace=True)
        exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)
        return exercise_df
    except FileNotFoundError:
        st.warning("Datasets not found. Using sample data.")
        return SAMPLE_DATA

exercise_df = load_datasets()

# Make prediction with enhanced visualization
st.header("🔥 Predicted Calories Burned")
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

if model:
    expected_features = ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_male"]
    df = df.reindex(columns=expected_features, fill_value=0)
    try:
        prediction = model.predict(df)
        confidence_score = model.score(df, prediction) if hasattr(model, 'score') else 0.85
        
        # Store prediction
        current_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prediction': float(prediction[0]),
            'confidence': float(confidence_score),
            'user_data': {
                'age': float(df['Age'].values[0]),
                'bmi': float(df['BMI'].values[0]),
                'duration': float(df['Duration'].values[0]),
                'heart_rate': float(df['Heart_Rate'].values[0]),
                'body_temp': float(df['Body_Temp'].values[0]),
                'gender': 'Male' if df['Gender_male'].values[0] == 1 else 'Female'
            }
        }
        
        st.session_state.user_history.append(current_data)
        save_to_json(st.session_state.user_history)
        
        # Update analysis data
        for key in st.session_state.analysis_data:
            if key == 'dates':
                st.session_state.analysis_data[key].append(current_data['timestamp'])
            elif key == 'calories':
                st.session_state.analysis_data[key].append(current_data['prediction'])
            else:
                st.session_state.analysis_data[key].append(current_data['user_data'][key])
        
        # Enhanced prediction display
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background-color: var(--card-bg); border-radius: 10px; border: 2px solid var(--primary);'>
                    <h2 style='color: var(--primary);'>Predicted Calories Burned</h2>
                    <h1 style='font-size: 4rem; margin: 1rem 0;'>{round(prediction[0], 2)}</h1>
                    <p style='font-size: 1.2rem; color: var(--text-secondary);'>kilocalories</p>
                    <div style='margin-top: 1rem; padding: 0.5rem; background-color: rgba(255, 75, 75, 0.1); border-radius: 5px;'>
                        <p style='margin: 0;'>Confidence: {confidence_score:.1%}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Quick insights
            st.markdown("""
                <div class='insight-card'>
                    <h4>💡 Quick Insights</h4>
                    <p>• This workout intensity is {} for your profile</p>
                    <p>• You're in the {} heart rate zone</p>
                    <p>• Estimated recovery time: {} minutes</p>
                </div>
            """.format(
                "optimal" if 100 <= df['Heart_Rate'].values[0] <= 140 else "sub-optimal",
                "fat-burning" if df['Heart_Rate'].values[0] < 120 else "cardio" if df['Heart_Rate'].values[0] < 160 else "peak",
                round(df['Duration'].values[0] * 0.5)
            ), unsafe_allow_html=True)
        
        # Enhanced visualizations
        st.header("📊 Detailed Analysis")
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Prediction History", "Similar Profiles", "Performance Insights"])
        
        with tab1:
            if len(st.session_state.prediction_history) > 1:
                history_df = pd.DataFrame(st.session_state.prediction_history)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df['prediction'],
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='#ff4b4b', width=3)
                ))
                fig.update_layout(
                    title="Your Calorie Burn Predictions Over Time",
                    xaxis_title="Time",
                    yaxis_title="Calories Burned",
                    plot_bgcolor='var(--card-bg)',
                    paper_bgcolor='var(--card-bg)',
                    font=dict(color='var(--text)'),
                    title_font_color='var(--primary)',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Start tracking your workouts to see your progress over time!")
        
        with tab2:
            if prediction is not None:
                def calculate_similarity(row):
                    weights = {'Age': 0.15, 'BMI': 0.35, 'Duration': 0.25, 'Heart_Rate': 0.20, 'Body_Temp': 0.05}
                    return sum(abs(row[col] - df[col].values[0]) * weights[col] for col in weights)
                
                exercise_df['similarity_score'] = exercise_df.apply(calculate_similarity, axis=1)
                similar_data = exercise_df.nsmallest(5, 'similarity_score')
                
                if not similar_data.empty:
                    display_columns = ['Age', 'Gender', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
                    similar_data = similar_data[display_columns].copy()
                    similar_data['Gender'] = similar_data['Gender'].map({1: 'Male', 0: 'Female'})
                    similar_data['BMI'] = similar_data['BMI'].round(2)
                    similar_data['Calories'] = similar_data['Calories'].round(2)
                    
                    # Enhanced similar profiles display
                    st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: var(--secondary);'>Similar Fitness Profiles</h3>
                            <p style='color: var(--text-secondary);'>These profiles have similar characteristics to your current workout:</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display the dataframe with enhanced styling
                    st.dataframe(
                        similar_data.style
                        .set_properties(**{
                            'background-color': 'var(--card-bg)',
                            'color': 'var(--text)',
                            'border-color': 'var(--primary)',
                            'padding': '0.5rem'
                        })
                        .set_table_styles([
                            {'selector': 'th', 'props': [
                                ('background-color', 'var(--primary)'),
                                ('color', 'white'),
                                ('font-weight', 'bold'),
                                ('padding', '0.5rem')
                            ]},
                            {'selector': 'td', 'props': [
                                ('border', '1px solid var(--primary)'),
                                ('padding', '0.5rem')
                            ]}
                        ])
                    )
                    
                    # Similar profiles visualization with enhanced colors
                    fig = px.scatter(similar_data, x='BMI', y='Calories',
                                   color='Duration', size='Heart_Rate',
                                   title='Similar Profiles Analysis',
                                   labels={'BMI': 'Body Mass Index', 'Calories': 'Calories Burned'},
                                   color_continuous_scale='Viridis')
                    fig.update_layout(
                        plot_bgcolor='var(--card-bg)',
                        paper_bgcolor='var(--card-bg)',
                        font=dict(color='var(--text)'),
                        title_font_color='var(--secondary)',
                        margin=dict(t=50, l=50, r=50, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No similar profiles found. Try adjusting your parameters.")
            else:
                st.info("Cannot find similar profiles without a valid prediction.")
        
        with tab3:
            if len(st.session_state.analysis_data['calories']) > 1:
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_calories = np.mean(st.session_state.analysis_data['calories'][:-1])
                    calorie_diff = prediction[0] - avg_calories
                    st.metric(
                        "Calorie Burn Trend",
                        f"{round(prediction[0], 1)} kcal",
                        f"{'+' if calorie_diff > 0 else ''}{round(calorie_diff, 1)} kcal"
                    )
                with col2:
                    duration_trend = np.polyfit(range(len(st.session_state.analysis_data['duration'])), 
                                             st.session_state.analysis_data['duration'], 1)[0]
                    st.metric(
                        "Duration Trend",
                        f"{round(df['Duration'].values[0], 1)} min",
                        f"{'+' if duration_trend > 0 else ''}{round(duration_trend, 1)} min/session"
                    )
                with col3:
                    heart_rate_trend = np.polyfit(range(len(st.session_state.analysis_data['heart_rate'])), 
                                               st.session_state.analysis_data['heart_rate'], 1)[0]
                    st.metric(
                        "Heart Rate Trend",
                        f"{round(df['Heart_Rate'].values[0], 1)} bpm",
                        f"{'+' if heart_rate_trend > 0 else ''}{round(heart_rate_trend, 1)} bpm/session"
                    )
                
                # Detailed insights
                st.markdown("""
                    <div class='insight-card'>
                        <h4>🎯 Performance Insights</h4>
                        <p>• Your current workout is {}% more intense than your average</p>
                        <p>• You're burning calories {} efficiently than before</p>
                        <p>• Your heart rate zone is {} for optimal fat burning</p>
                    </div>
                """.format(
                    round((df['Heart_Rate'].values[0] / np.mean(st.session_state.analysis_data['heart_rate']) - 1) * 100),
                    "more" if calorie_diff > 0 else "less",
                    "optimal" if 120 <= df['Heart_Rate'].values[0] <= 140 else "sub-optimal"
                ), unsafe_allow_html=True)
            else:
                st.info("Complete more workouts to see detailed performance insights!")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.warning("Prediction unavailable due to missing model.")

# Enhanced footer
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 2rem; background-color: var(--card-bg); border-radius: 10px; margin-top: 2rem; border: 1px solid var(--primary);'>
        <h3 style='color: var(--primary);'>🚀 Stay Fit & Track Your Progress Daily! 💪</h3>
        <p style='color: var(--text-secondary); margin-top: 1rem;'>Powered by advanced AI algorithms for accurate predictions</p>
    </div>
""", unsafe_allow_html=True)
st.balloons()