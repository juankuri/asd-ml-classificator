import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/final_model.pkl")
columns = joblib.load("models/columns.pkl")

st.title("Autism Screening Predictor")

data = {}

# Inputs
for i in range(1, 11):
    data[f"A{i}_Score"] = st.selectbox(f"A{i}_Score", [0, 1])

data["age"] = st.number_input("Age", 1, 100)
data["gender"] = st.selectbox("Gender", ["Male", "Female"])
data["ethnicity"] = st.text_input("Ethnicity")
data["jaundice"] = st.selectbox("Jaundice", ["yes", "no"])
data["austim"] = st.selectbox("Family autism", ["yes", "no"])
data["contry_of_res"] = st.text_input("Country")
data["used_app_before"] = st.selectbox("Used app before", ["yes", "no"])
data["relation"] = st.selectbox("Relation", ["Self", "Parent", "Relative", "Others"])

if st.button("Predict"):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    threshold = joblib.load("models/threshold.pkl")

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    pred = 1 if prob > threshold else 0

    

    st.subheader("Resultado")
    
    if prob < 0.3:
        st.success("Riesgo bajo")
    elif prob < 0.6:
        st.warning("Riesgo moderado")
    else:
        st.error("Riesgo alto") 
    
    st.write("Autismo:", "Sí" if pred == 1 else "No")
    st.write("Probabilidad:", prob)
