

# import streamlit as st
# import pandas as pd
# from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# st.set_page_config(page_title="Bank Conversion Predictor", page_icon="📈", layout="wide")

# st.title("📊 Bank Marketing Conversion Predictor")
# st.write("Predict whether a customer will subscribe to a term deposit.")

# st.divider()

# # -----------------------------
# # Category Options (Dataset Based)
# # -----------------------------

# occupation_options = [
# "administrative_staff","jobless","retired_worker","business_owner",
# "manual_worker","student","technical_specialist","executive",
# "service_worker","independent_worker","domestic_worker","Other"
# ]

# education_options = [
# "high_school","college","elementary_school","Other"
# ]

# marital_options = [
# "married","single","divorced","Other"
# ]

# communication_options = [
# "mobile","landline","Other"
# ]

# month_options = [
# "January","February","March","April","May","June",
# "July","August","September","October","November","December","Other"
# ]

# previous_outcome_options = [
# "successful","unsuccessful","other_outcome","Other"
# ]

# # -----------------------------
# # Input Form
# # -----------------------------

# with st.form("prediction_form"):

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.slider("Age",18,95,35)

#         occupation = st.selectbox(
#             "Occupation",
#             occupation_options
#         )

#         education_level = st.selectbox(
#             "Education Level",
#             education_options
#         )

#         marital_status = st.selectbox(
#             "Marital Status",
#             marital_options
#         )

#     with col2:

#         communication_channel = st.selectbox(
#             "Communication Channel",
#             communication_options
#         )

#         call_day = st.slider("Day of Month Contacted",1,31,10)

#         call_month = st.selectbox(
#             "Call Month",
#             month_options
#         )

#     with col3:
#         call_duration = st.slider("Call Duration (seconds)",1,3000,200)

#         call_frequency = st.slider("Number of Calls",1,60,2)

#         previous_campaign_outcome = st.selectbox(
#             "Previous Campaign Outcome",
#             previous_outcome_options
#         )

#     submitted = st.form_submit_button("Predict Conversion")

# # -----------------------------
# # Convert "Other" → unidentified
# # -----------------------------

# def handle_other(value):
#     if value == "Other":
#         return "unidentified"
#     return value

# # -----------------------------
# # Prediction
# # -----------------------------

# if submitted:

#     occupation = handle_other(occupation)
#     education_level = handle_other(education_level)
#     marital_status = handle_other(marital_status)
#     communication_channel = handle_other(communication_channel)
#     call_month = handle_other(call_month)
#     previous_campaign_outcome = handle_other(previous_campaign_outcome)

#     data = CustomData(
#         age=age,
#         occupation=occupation,
#         marital_status=marital_status,
#         education_level=education_level,
#         communication_channel=communication_channel,
#         call_day=call_day,
#         call_month=call_month,
#         call_duration=call_duration,
#         call_frequency=call_frequency,
#         previous_campaign_outcome=previous_campaign_outcome
#     )

#     df = data.get_data_as_dataframe()

#     # pipeline = PredictPipeline()

#     # prediction = pipeline.predict(df)[0]

#     # st.divider()

#     # if prediction == "converted":
#     #     st.success("✅ Customer is likely to subscribe to the term deposit.")
#     # else:
#     #     st.error("❌ Customer is unlikely to subscribe.")

#     # st.subheader("Input Summary")

#     # st.dataframe(df)
    
#     pipeline = PredictPipeline()

#     prediction, probability = pipeline.predict(df)

#     score = round(probability * 100,2)

#     st.divider()

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric("Prediction", prediction.upper())

#     with col2:
#         st.metric("Conversion Probability", f"{probability:.2%}")

#     with col3:
#         st.metric("Conversion Score", f"{score}/100")

#     st.subheader("Conversion Likelihood")

#     st.progress(score/100)

#     if prediction == "converted":
#         st.success("Customer is highly likely to subscribe.")
#     else:
#         st.warning("Customer is unlikely to subscribe.")

#     st.subheader("Input Summary")
#     st.dataframe(df)
    
    
    
    
import pandas as pd
from flask import Flask, request, jsonify

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

# Load pipeline once
pipeline = PredictPipeline()


@app.route("/")
def home():
    return {
        "message": "Bank Marketing Conversion Prediction API is running"
    }


# ---------------------------------------------------------
# Prediction API
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        input_data = CustomData(
            age=data.get("age"),
            occupation=data.get("occupation"),
            marital_status=data.get("marital_status"),
            education_level=data.get("education_level"),
            balance=data.get("balance"),
            housing=data.get("housing", "no"),
            loan=data.get("loan", "no"),
            communication_channel=data.get("communication_channel"),
            call_day=data.get("call_day"),
            call_month=data.get("call_month"),
            call_duration=data.get("call_duration"),
            call_frequency=data.get("call_frequency"),
            previous_campaign_outcome=data.get("previous_campaign_outcome")
        )

        df = input_data.get_data_as_dataframe()

        prediction, probability = pipeline.predict(df)

        result = {
            "prediction": prediction,
            "conversion_probability": round(probability, 4),
            "conversion_score": round(probability * 100, 2)
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


# ---------------------------------------------------------
# Run server
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )