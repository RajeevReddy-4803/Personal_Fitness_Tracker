# 🏋️‍♂️ Personal Fitness Tracker  
An AI-powered **calorie burn prediction** system that leverages **machine learning** to provide real-time fitness insights based on user inputs.

## 🚀 Features  
- 🔥 **Calorie Prediction**: Estimates calories burned based on Age, BMI, Duration, Heart Rate, Body Temperature, and Gender.  
- 📊 **ML Models Used**:  
  - Random Forest (Tuned & Untuned)  
  - Linear Regression (Tuned with Polynomial Features)  
- 🎯 **Best Model Performance**:  
  - **Untuned Random Forest** (RMSE: `3.96`, MAE: `2.58`, MSE: `15.66`)  
- 🌍 **Interactive Web App**: Users can input their details and get real-time predictions.  
- 📈 **Comparative Insights**: See how your stats compare to other users (e.g., "Your exercise duration is longer than 98.9% of users").  

---

## ⚙️ Installation  

### 1️⃣ **Clone the Repository**  

git clone https://github.com/YOUR_USERNAME/personal-fitness-tracker.git
cd personal-fitness-tracker 

### 2️⃣ **Create and Activate a Virtual Environment**
python -m venv venv  # Create a virtual environment  
source venv/bin/activate  # On macOS/Linux  
venv\Scripts\activate  # On Windows  

### 3️⃣ **Install Dependencies**
pip install -r requirements.txt

### 4️⃣ **Run the Application**
streamlit run app.py  # If using Streamlit for the frontend

---

📌 Usage Guide
Open the web interface.
Input your Age, BMI, Duration, Heart Rate, Body Temperature, and Gender.
Click "Predict" to get the estimated calories burned.
View comparative insights and make informed fitness decisions!

---

🛠️ Tech Stack
Machine Learning: Scikit-Learn, Pandas, NumPy, Matplotlib
Web Framework: Streamlit (for interactive UI)
Backend: Python
Version Control: Git & GitHub

----


