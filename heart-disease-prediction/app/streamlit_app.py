import streamlit as st
import pandas as pd
import joblib

model = joblib.load('../models/best_model.joblib')

st.title("❤️ Heart Disease Prediction App")
st.sidebar.header("Enter Patient Details")

def user_input():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (1=Male, 0=Female)', [1, 0])
    cp = st.sidebar.selectbox('Chest Pain Type (0–3)', [0,1,2,3])
    trestbps = st.sidebar.slider('Resting BP', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar >120 mg/dl', [1,0])
    restecg = st.sidebar.selectbox('Resting ECG (0–2)', [0,1,2])
    thalach = st.sidebar.slider('Max Heart Rate', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise Angina (1=Yes, 0=No)', [1,0])
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.5, 1.0)
    slope = st.sidebar.selectbox('Slope (0–2)', [0,1,2])
    ca = st.sidebar.selectbox('Major Vessels (0–4)', [0,1,2,3,4])
    thal = st.sidebar.selectbox('Thal (0–3)', [0,1,2,3])

    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    return pd.DataFrame(data, index=[0])

input_df = user_input()
prediction = model.predict(input_df)
prob = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob:.2f})")
else:
    st.success(f" Low Risk of Heart Disease (Probability: {prob:.2f})")
