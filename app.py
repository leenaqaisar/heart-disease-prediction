from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

st.title("Heart Disease Prediction System")
st.info("Educational demonstration only—not a medical diagnosis.")


@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "heart_disease_model.pkl"

    with model_path.open("rb") as file:
        return pickle.load(file)


model = load_model()


with st.form("patient_form"):
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age (years)", 18, 100, 63)

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: {0: "Female", 1: "Male"}[x]
        )

        cp = st.selectbox(
            "Chest pain type",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical angina",
                2: "Atypical angina",
                3: "Non-anginal pain",
                4: "Asymptomatic"
            }[x]
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)", 60, 260, 145
        )

        chol = st.number_input(
            "Cholesterol (mg/dL)", 80, 700, 233
        )

        fbs = st.selectbox(
            "Fasting blood sugar >120 mg/dL?",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T abnormality",
                2: "Probable/definite LV hypertrophy"
            }[x]
        )

    with right:
        thalach = st.number_input(
            "Maximum achieved heart rate (bpm)", 50, 250, 150
        )

        exang = st.selectbox(
            "Exercise-induced angina?",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )

        oldpeak = st.number_input(
            "ST depression (oldpeak)", 0.0, 10.0, 2.3, step=0.1
        )

        slope = st.selectbox(
            "Peak exercise ST slope",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping"
            }[x]
        )

        ca = st.selectbox(
            "Major vessels coloured by fluoroscopy", [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thal test result",
            [3, 6, 7],
            format_func=lambda x: {
                3: "Normal",
                6: "Fixed defect",
                7: "Reversible defect"
            }[x]
        )

    submitted = st.form_submit_button("Predict")


if submitted:
    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = int(model.predict(patient)[0])

    if prediction == 1:
        st.subheader("Predicted label: Disease present")
    else:
        st.subheader("Predicted label: Disease absent")

    st.caption("A model prediction cannot confirm or exclude disease.")


st.markdown(
    "[Dataset source: UCI Heart Disease]"
    "(https://archive.ics.uci.edu/dataset/45/heart+disease)"
)