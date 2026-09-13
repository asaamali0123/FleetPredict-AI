import streamlit as st
import pandas as pd

from src.schema_validator import validate_sensor_dataframe


st.set_page_config(
    page_title="FleetPredict AI",
    page_icon="🚛",
    layout="wide",
)


# ---------------------------------------------------------
# Page Header
# ---------------------------------------------------------
st.title("🚛 FleetPredict AI")
st.subheader("Fleet Health & Predictive Maintenance Dashboard")

st.write(
    "Upload vehicle sensor telemetry to validate your fleet data "
    "and begin the analysis."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("FleetPredict AI")
st.sidebar.caption("Sensor Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sensor CSV",
    type=["csv"],
)


# ---------------------------------------------------------
# Main Content
# ---------------------------------------------------------
if uploaded_file is None:
    st.info(
        "Please upload a sensor CSV file from the sidebar to begin."
    )

else:
    st.subheader("Sensor Data Validation")

    try:
        dataframe = pd.read_csv(uploaded_file)

    except Exception as error:
        st.error(
            f"Could not read the uploaded CSV file: {error}"
        )
        st.stop()

    # Check for empty CSV
    if dataframe.empty:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    # Use the existing backend validator
    validation_result = validate_sensor_dataframe(dataframe)

    if validation_result["valid"]:
        st.success("Sensor CSV validation passed.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Records",
                validation_result["row_count"],
            )

        with col2:
            st.metric(
                "Vehicles",
                validation_result["vehicle_count"],
            )

        st.write("### Uploaded Data Preview")
        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
        )

        if validation_result["warnings"]:
            st.warning("Validation warnings:")

            for warning in validation_result["warnings"]:
                st.write(f"- {warning}")

    else:
        st.error("Sensor CSV validation failed.")

        if validation_result["errors"]:
            st.write("### Validation Errors")

            for error in validation_result["errors"]:
                st.write(f"- {error}")

        if validation_result["warnings"]:
            st.write("### Validation Warnings")

            for warning in validation_result["warnings"]:
                st.write(f"- {warning}")