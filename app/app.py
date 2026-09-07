import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://api:8000"
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to estimate the probability "
    "of customer churn."
)


st.subheader("Customer Information")


col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )


with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


with col3:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


st.divider()


if st.button(
    "Predict Churn",
    type="primary"
):

    customer_data = {

        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            probability = result[
                "churn_probability"
            ]

            prediction = result[
                "churn_prediction"
            ]

            label = result[
                "prediction_label"
            ]

            threshold = result[
                "classification_threshold"
            ]

            st.subheader("Prediction Result")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Churn Probability",
                    f"{probability:.2%}"
                )

            with metric2:
                st.metric(
                    "Prediction",
                    label
                )

            with metric3:
                st.metric(
                    "Threshold",
                    f"{threshold:.2f}"
                )

            if prediction == 1:

                st.error(
                    "⚠️ This customer is predicted "
                    "to be likely to churn."
                )

            else:

                st.success(
                    "✅ This customer is predicted "
                    "to be likely to stay."
                )

        else:

            st.error(
                f"API request failed: "
                f"{response.status_code}"
            )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Unable to connect to the prediction API: "
            f"{error}"
        )