

import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

st.set_page_config(page_title="Bank Conversion Predictor", page_icon="📈", layout="wide")

st.title("📊 Bank Marketing Conversion Predictor")
st.write("Predict whether a customer will subscribe to a term deposit.")

st.divider()

# -----------------------------
# Category Options (Dataset Based)
# -----------------------------

occupation_options = [
"administrative_staff","jobless","retired_worker","business_owner",
"manual_worker","student","technical_specialist","executive",
"service_worker","independent_worker","domestic_worker","Other"
]

education_options = [
"high_school","college","elementary_school","Other"
]

marital_options = [
"married","single","divorced","Other"
]

communication_options = [
"mobile","landline","Other"
]

month_options = [
"January","February","March","April","May","June",
"July","August","September","October","November","December","Other"
]

previous_outcome_options = [
"successful","unsuccessful","other_outcome","Other"
]

# -----------------------------
# Input Form
# -----------------------------

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age",18,95,35)

        occupation = st.selectbox(
            "Occupation",
            occupation_options
        )

        education_level = st.selectbox(
            "Education Level",
            education_options
        )

        marital_status = st.selectbox(
            "Marital Status",
            marital_options
        )

    with col2:

        communication_channel = st.selectbox(
            "Communication Channel",
            communication_options
        )

        call_day = st.slider("Day of Month Contacted",1,31,10)

        call_month = st.selectbox(
            "Call Month",
            month_options
        )

    with col3:
        call_duration = st.slider("Call Duration (seconds)",1,3000,200)

        call_frequency = st.slider("Number of Calls",1,60,2)

        previous_campaign_outcome = st.selectbox(
            "Previous Campaign Outcome",
            previous_outcome_options
        )

    submitted = st.form_submit_button("Predict Conversion")

# -----------------------------
# Convert "Other" → unidentified
# -----------------------------

def handle_other(value):
    if value == "Other":
        return "unidentified"
    return value

# -----------------------------
# Prediction
# -----------------------------

if submitted:

    occupation = handle_other(occupation)
    education_level = handle_other(education_level)
    marital_status = handle_other(marital_status)
    communication_channel = handle_other(communication_channel)
    call_month = handle_other(call_month)
    previous_campaign_outcome = handle_other(previous_campaign_outcome)

    data = CustomData(
        age=age,
        occupation=occupation,
        marital_status=marital_status,
        education_level=education_level,
        communication_channel=communication_channel,
        call_day=call_day,
        call_month=call_month,
        call_duration=call_duration,
        call_frequency=call_frequency,
        previous_campaign_outcome=previous_campaign_outcome
    )

    df = data.get_data_as_dataframe()

    pipeline = PredictPipeline()

    prediction = pipeline.predict(df)[0]

    st.divider()

    if prediction == "converted":
        st.success("✅ Customer is likely to subscribe to the term deposit.")
    else:
        st.error("❌ Customer is unlikely to subscribe.")

    st.subheader("Input Summary")

    st.dataframe(df)