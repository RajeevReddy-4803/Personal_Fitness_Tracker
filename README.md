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

# Personal Fitness Tracker

A modern, AI-powered fitness tracking application that helps users monitor their workouts and predict calorie burn with precision.

## 🌟 Features

- **Smart Calorie Prediction**: AI-powered calorie burn prediction based on personal metrics
- **Interactive Dashboard**: Beautiful and intuitive user interface with real-time updates
- **Detailed Analytics**: Comprehensive workout analysis and progress tracking
- **Similar Profiles**: Compare your workout patterns with similar fitness profiles
- **Performance Insights**: Track your progress and get personalized recommendations

## 🎨 UI Enhancements

- Modern color scheme with primary green, secondary blue, and accent orange
- Responsive design with card-based layout
- Interactive visualizations and charts
- Enhanced data presentation with styled tables
- Smooth animations and transitions
- Professional typography and spacing

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly
- **Data Storage**: JSON

## 📋 Requirements

```
streamlit>=1.24.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.0.0
plotly>=5.13.0
joblib>=1.2.0
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/personal-fitness-tracker.git
cd personal-fitness-tracker
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## 📊 Data Structure

The application uses two main datasets:
- `exercise.csv`: Contains workout metrics
- `calories.csv`: Contains calorie burn data

## 🤖 Model Information

- Trained using Random Forest algorithm
- Features: Age, BMI, Duration, Heart Rate, Body Temperature, Gender
- Model file: `trained_model.pkl`
- Scaler file: `scaler.pk1`

## 📝 Usage

1. Enter your personal metrics in the sidebar
2. View your predicted calorie burn
3. Explore similar fitness profiles
4. Track your progress over time
5. Get personalized insights

## 🎯 Features in Detail

### Smart Prediction
- Real-time calorie burn prediction
- Confidence score for predictions
- Input validation and feedback

### Analytics Dashboard
- Interactive charts and graphs
- Performance trends
- Workout intensity analysis

### Similar Profiles
- Profile matching algorithm
- Comparative analysis
- Visual representation of similar workouts

### Performance Tracking
- Historical data visualization
- Progress metrics
- Personalized recommendations

## 🔒 Data Privacy

- All data is stored locally
- No external data transmission
- Secure JSON storage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Scikit-learn for the machine learning capabilities
- Plotly for the visualization tools

----


