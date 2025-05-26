# Data Structure Documentation

## calories.csv
Expected columns:
- User_ID (int): Unique identifier for each user
- Calories (float): Calories burned during exercise

## exercise.csv
Expected columns:
- User_ID (int): Unique identifier for each user
- Age (int): Age of the user
- Gender (int): 1 for male, 0 for female
- Height (float): Height in centimeters
- Weight (float): Weight in kilograms
- Duration (int): Exercise duration in minutes
- Heart_Rate (int): Average heart rate during exercise
- Body_Temp (float): Body temperature in Celsius

## Example Data Format

### calories.csv
```csv
User_ID,Calories
1,250.5
2,300.0
3,275.8
```

### exercise.csv
```csv
User_ID,Age,Gender,Height,Weight,Duration,Heart_Rate,Body_Temp
1,25,1,175.0,70.5,30,120,37.5
2,30,0,165.0,60.0,45,130,37.8
3,35,1,180.0,80.0,20,110,37.2
```

## Data Validation Rules

1. Age: 10-100 years
2. BMI: 15.0-40.0
3. Duration: 0-60 minutes
4. Heart Rate: 60-180 bpm
5. Body Temperature: 36.0-40.0 °C
6. Gender: 0 (Female) or 1 (Male)

## Data Preprocessing

1. BMI is calculated as: Weight (kg) / (Height (m))²
2. All numerical values should be rounded to 2 decimal places
3. Missing values should be handled appropriately
4. Outliers should be removed or capped at reasonable limits 