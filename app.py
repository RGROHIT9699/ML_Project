
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

st.markdown(
    "<h1 style='text-align: center; color: #3A3A3A; background-color: #E3F2FD; font-size: 40px; font-family: Verdana;'>Diabetes Prediction App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #5F6368; font-size: 18px; font-family: Arial;'>This app predicts whether a patient is diabetic based on their health data.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

st.sidebar.header('<span style="color:#3A3A3A; font-family:Arial; font-size:20px;">Enter Patient Data</span>', unsafe_allow_html=True)
st.sidebar.write("<span style='color:#5F6368; font-family: Arial;'>Please provide the following details for a diabetes checkup:</span>", unsafe_allow_html=True)

def calc():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    report_data = pd.DataFrame(output, index=[0])
    return report_data

user_data = calc()
st.subheader("<span style='font-size: 24px; color: #333333;'>Patient Data Summary</span>", unsafe_allow_html=True)
st.write(user_data)

x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

progress = st.progress(0)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
progress.progress(100)

result = rf.predict(user_data)

output = 'You are not Diabetic' if result[0] == 0 else 'You are Diabetic'
st.markdown(
    f"<h2 style='text-align: center; color: {'#388E3C' if result[0] == 0 else '#D32F2F'}; background-color: #E3F2FD; font-size: 30px; font-family: Arial;'>{output}</h2>",
    unsafe_allow_html=True
)

accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.subheader("<span style='font-size: 20px; color: #333333;'>Model Accuracy</span>", unsafe_allow_html=True)
st.write(f"<span style='font-size:18px;'>{accuracy:.2f}%</span>", unsafe_allow_html=True)
