import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("alz_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Alzheimer’s Prediction System", layout="wide")

st.title("🧠 Alzheimer’s Disease Prediction System")
st.write("Enter patient health & lifestyle details to predict Alzheimer’s risk")

# Sidebar input
st.sidebar.header("Patient Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 60, 90, 70)
    gender = st.sidebar.selectbox("Gender", (0,1))
    ethnicity = st.sidebar.selectbox("Ethnicity", (0,1,2,3))
    education = st.sidebar.selectbox("Education Level", (0,1,2,3))
    bmi = st.sidebar.slider("BMI", 15, 40, 25)
    smoking = st.sidebar.selectbox("Smoking", (0,1))
    alcohol = st.sidebar.slider("Alcohol Consumption / Week", 0,20,2)
    activity = st.sidebar.slider("Physical Activity (hours/week)", 0,10,3)
    diet = st.sidebar.slider("Diet Quality Score", 0,10,6)
    sleep = st.sidebar.slider("Sleep Quality", 4,10,7)

    family = st.sidebar.selectbox("Family History of Alzheimer’s", (0,1))
    cardio = st.sidebar.selectbox("Cardiovascular Disease", (0,1))
    diabetes = st.sidebar.selectbox("Diabetes", (0,1))
    depression = st.sidebar.selectbox("Depression", (0,1))
    injury = st.sidebar.selectbox("Head Injury", (0,1))
    hypertension = st.sidebar.selectbox("Hypertension", (0,1))

    systolic = st.sidebar.slider("Systolic BP", 90,180,130)
    diastolic = st.sidebar.slider("Diastolic BP", 60,120,80)
    chol_total = st.sidebar.slider("Cholesterol Total", 150,300,200)
    chol_ldl = st.sidebar.slider("LDL", 50,200,100)
    chol_hdl = st.sidebar.slider("HDL", 20,100,50)
    chol_tri = st.sidebar.slider("Triglycerides", 50,400,150)

    mmse = st.sidebar.slider("MMSE Score", 0,30,20)
    fa = st.sidebar.slider("Functional Assessment", 0,10,7)
    memory = st.sidebar.selectbox("Memory Complaints", (0,1))
    behavior = st.sidebar.selectbox("Behavioral Problems", (0,1))
    adl = st.sidebar.slider("Daily Living Score (ADL)", 0,10,6)

    confusion = st.sidebar.selectbox("Confusion", (0,1))
    disorientation = st.sidebar.selectbox("Disorientation", (0,1))
    personality = st.sidebar.selectbox("Personality Changes", (0,1))
    tasks = st.sidebar.selectbox("Difficulty Completing Tasks", (0,1))
    forget = st.sidebar.selectbox("Forgetfulness", (0,1))

    data = np.array([
        age, gender, ethnicity, education, bmi, smoking, alcohol, activity, diet, sleep,
        family, cardio, diabetes, depression, injury, hypertension, systolic, diastolic,
        chol_total, chol_ldl, chol_hdl, chol_tri, mmse, fa, memory, behavior, adl,
        confusion, disorientation, personality, tasks, forget
    ])
    
    return data

data = user_input_features()

# Prediction
if st.button("Predict Alzheimer's Risk"):
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠ High Risk of Alzheimer’s Disease.\nProbability: {prob:.2f}")
    else:
        st.success(f"🟢 Low Risk of Alzheimer’s Disease.\nProbability: {prob:.2f}")

