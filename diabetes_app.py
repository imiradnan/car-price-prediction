import streamlit as st
import pickle
import numpy as np


model = pickle.load(
    open(
        "diabetes_model.pkl",
        "rb"
    )
)


st.title("🩺 Diabetes Risk Prediction Dashboard")

st.write(
    "Predict diabetes risk using health parameters"
)


preg = st.number_input(
    "Pregnancies",
    value=1
)

glucose = st.number_input(
    "Glucose Level",
    value=120
)

bp = st.number_input(
    "Blood Pressure",
    value=80
)

skin = st.number_input(
    "Skin Thickness",
    value=20
)

insulin = st.number_input(
    "Insulin",
    value=80
)

bmi = st.number_input(
    "BMI",
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    value=0.5
)

age = st.number_input(
    "Age",
    value=30
)


if st.button("Predict"):

    data = np.array(
        [[
            preg,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]]
    )


    prediction = model.predict(data)

    probability = model.predict_proba(data)


    if prediction[0] == 1:

        st.error(
            "High Diabetes Risk ⚠️"
        )

    else:

        st.success(
            "Low Diabetes Risk ✅"
        )


    st.write(
        "Risk Probability:",
        round(
            probability[0][1],
            2
        )
    )