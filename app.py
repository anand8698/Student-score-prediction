import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = '/Users/anandtripathi/Desktop/final project/student_performance_data_updated_v2.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data['Performance_Category'] = data['Performance_Category'].map({
    'High-Performing': 2, 'Average': 1, 'At-Risk': 0
})

X = data.drop(columns=['Student_ID', 'Performance_Category'])
y = data['Performance_Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define categories globally
categories = {0: "At-Risk", 1: "Average", 2: "High-Performing"}

# Streamlit App
st.title("Student Performance Prediction App")

# Input fields
attendance = st.slider("Attendance (%)", min_value=50, max_value=100, value=75)
past_grades = st.slider("Past Grades (0-100)", min_value=40, max_value=100, value=70)
hours_studied = st.slider("Hours Studied per Week", min_value=5, max_value=25, value=15)
medical_emergency = st.selectbox("Medical Emergency (Yes/No)", options=['No', 'Yes'])
assignment_score = st.slider("Class Assignment Score (out of 10)", min_value=5, max_value=10, value=8)

# Convert Medical Emergency to binary
medical_emergency = 1 if medical_emergency == 'Yes' else 0

# Prepare input for prediction
input_data = pd.DataFrame([[attendance, past_grades, hours_studied, medical_emergency, assignment_score]],
                          columns=['Attendance', 'Past_Grades', 'Hours_Studied_per_Week', 'Medical_Emergency', 'Class_Assignment_Score'])

# Prediction and output
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Performance Category: **{categories[prediction]}**")

# Model Performance (Optional)
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=categories.values())
    st.write("Model Accuracy:", round(accuracy * 100, 2), "%")
    st.text("Classification Report:\n" + report)