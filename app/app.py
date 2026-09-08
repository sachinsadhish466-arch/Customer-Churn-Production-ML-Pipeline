import os

import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# Header
# =========================================================

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    **Production ML Pipeline Demo**

    Enter customer information below to estimate the customer's
    churn probability using the deployed machine learning model.
    """
)


# =========================================================
# API Status
# =========================================================

with st.sidebar:

    st.header("System Status")

    st.caption(
        f"Prediction API: `{API_URL}`"
    )

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if health_response.status_code == 200:

            st.success("API Online")

        else:

            st.warning(
                f"API returned status "
                f"{health_response.status_code}"
            )

    except requests.exceptions.RequestException:

        st.error("API Unavailable")


# =========================================================
# Customer Information
# =========================================================

st.subheader("Customer Information")

st.caption(
    "Provide the customer's current account and service details."
)


col1, col2, col3 = st.columns(3)


# =========================================================
# Column 1 — Customer Profile
# =========================================================

with col1:

    st.markdown("#### Customer Profile")

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda value: (
            "Yes" if value == 1 else "No"
        )
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
        value=12,
        step=1
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )


# =========================================================
# Column 2 — Internet & Services
# =========================================================

with col2:

    st.markdown("#### Internet & Services")

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# =========================================================
# Column 3 — Billing Information
# =========================================================

with col3:

    st.markdown("#### Billing Information")

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
        [
            "Yes",
            "No"
        ]
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
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


# =========================================================
# Prediction Button
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    type="primary",
    use_container_width=True
)


# =========================================================
# Prediction
# =========================================================

if predict_button:

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

        with st.spinner(
            "Running prediction..."
        ):

            response = requests.post(
                f"{API_URL}/predict",
                json=customer_data,
                timeout=30
            )


        # =================================================
        # Successful Prediction
        # =================================================

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


            # =============================================
            # Prediction Result
            # =============================================

            st.subheader(
                "Prediction Result"
            )


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
                    "Classification Threshold",
                    f"{threshold:.2f}"
                )


            # =============================================
            # Risk Assessment
            # =============================================

            st.markdown(
                "### Risk Assessment"
            )

            st.progress(
                probability
            )


            if prediction == 1:

                st.error(
                    "⚠️ **High Churn Risk** — "
                    "the model predicts that this customer "
                    "is likely to churn."
                )

                st.info(
                    "Consider reviewing this customer's "
                    "contract, service usage, and billing "
                    "profile for possible retention actions."
                )

            else:

                st.success(
                    "✅ **Lower Churn Risk** — "
                    "the model predicts that this customer "
                    "is likely to stay."
                )


            # =============================================
            # Prediction Details
            # =============================================

            with st.expander(
                "View Prediction Details"
            ):

                st.json(
                    result
                )


        # =================================================
        # API Error
        # =================================================

        else:

            st.error(
                f"API request failed with status "
                f"{response.status_code}: "
                f"{response.text}"
            )


    # =====================================================
    # Connection Error
    # =====================================================

    except requests.exceptions.RequestException as error:

        st.error(
            "Unable to connect to the prediction API."
        )

        st.caption(
            f"Connection details: {error}"
        )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Customer Churn Prediction • "
    "FastAPI + XGBoost + Docker + AWS ECS Fargate"
)