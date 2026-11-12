
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sklearn
from ucimlrepo import fetch_ucirepo

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    logistic_model = joblib.load("logistic_model.joblib")
    random_forest_model = joblib.load("random_forest_model.joblib")
    return logistic_model, random_forest_model


logistic_model, random_forest_model = load_models()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():
    ai4i = fetch_ucirepo(id=601)

    features = ai4i.data.features
    targets = ai4i.data.targets

    dataset = pd.concat([features, targets], axis=1)

    return dataset


try:
    df = load_dataset()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("⚙️ Predictive Maintenance Dashboard")

st.markdown(
    """
    ### AI-Based Machine Failure Prediction

    An interactive dashboard for exploring machine data,
    comparing machine-learning models, and predicting
    potential machine failures.
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Data Explorer",
        "Model Performance",
        "Failure Prediction",
        "Feature Importance"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Predictive Maintenance using Machine Learning"
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Dashboard Overview")

    total_records = len(df)

    failure_count = int(df["Machine failure"].sum())

    normal_count = total_records - failure_count

    failure_rate = (failure_count / total_records) * 100

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Machines", f"{total_records:,}")
    col2.metric("Normal Records", f"{normal_count:,}")
    col3.metric("Failure Records", f"{failure_count:,}")
    col4.metric("Failure Rate", f"{failure_rate:.2f}%")

    st.divider()

    # FAILURE DISTRIBUTION

    st.subheader("Machine Failure Distribution")

    distribution = pd.DataFrame({
        "Condition": ["Normal", "Failure"],
        "Count": [normal_count, failure_count]
    })

    fig = px.pie(
        distribution,
        names="Condition",
        values="Count",
        hole=0.45,
        title="Normal vs Failure Records"
    )

    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count: %{value:,}<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset:** UCI AI4I 2020 Predictive Maintenance")
        st.write(f"**Number of records:** {total_records:,}")
        st.write(f"**Number of columns:** {len(df.columns)}")

    with col2:
        st.write("**Target variable:** Machine failure")
        st.write("**Normal class:** 0")
        st.write("**Failure class:** 1")

# --------------------------------------------------
# OTHER SECTIONS - TO BE DEVELOPED
# --------------------------------------------------


elif page == "Data Explorer":

    st.header("Data Explorer")

    st.markdown(
        "Explore, filter, analyze and download the AI4I 2020 dataset."
    )

    # DATASET SUMMARY
    st.subheader("Dataset Preview")

    st.write(f"Total records: {len(df):,}")

    # FILTERS
    col1, col2 = st.columns(2)

    with col1:
        machine_types = st.multiselect(
            "Select Machine Type",
            options=sorted(df["Type"].unique()),
            default=sorted(df["Type"].unique())
        )

    with col2:
        failure_filter = st.selectbox(
            "Machine Condition",
            options=["All", "Normal", "Failure"]
        )

    # APPLY FILTERS
    filtered_df = df[df["Type"].isin(machine_types)]

    if failure_filter == "Normal":
        filtered_df = filtered_df[
            filtered_df["Machine failure"] == 0
        ]

    elif failure_filter == "Failure":
        filtered_df = filtered_df[
            filtered_df["Machine failure"] == 1
        ]

    # DISPLAY FILTERED DATA
    st.subheader("Filtered Machine Records")

    st.write(f"Records found: {len(filtered_df):,}")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

    # DOWNLOAD
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset",
        data=csv_data,
        file_name="filtered_machine_data.csv",
        mime="text/csv"
    )

    st.divider()

    # DESCRIPTIVE STATISTICS
    st.subheader("Descriptive Statistics")

    st.dataframe(
        filtered_df.describe(),
        use_container_width=True
    )



elif page == "Model Performance":

    st.header("Model Performance Comparison")

    st.markdown(
        "Evaluation of Logistic Regression and Random Forest "
        "on the held-out test dataset."
    )

    # --------------------------------------------------
    # MODEL EVALUATION RESULTS
    # --------------------------------------------------

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Logistic Regression": [82.45, 14.18, 82.35, 24.19],
        "Random Forest": [97.90, 90.62, 42.65, 58.00]
    })

    # METRIC COMPARISON
    st.subheader("Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Logistic Regression**")
        st.metric("Accuracy", "82.45%")
        st.metric("Precision", "14.18%")
        st.metric("Recall", "82.35%")
        st.metric("F1-Score", "24.19%")

    with col2:
        st.markdown("**Random Forest**")
        st.metric("Accuracy", "97.90%")
        st.metric("Precision", "90.62%")
        st.metric("Recall", "42.65%")
        st.metric("F1-Score", "58.00%")

    st.divider()

    # COMPARISON CHART
    st.subheader("Model Metrics Comparison")

    metrics_melted = metrics.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Score"
    )

    fig = px.bar(
        metrics_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        title="Logistic Regression vs Random Forest"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Score (%)",
        yaxis_range=[0, 110]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------

    st.subheader("Confusion Matrices")

    cm1 = np.array([
        [1593, 339],
        [12, 56]
    ])

    cm2 = np.array([
        [1929, 3],
        [39, 29]
    ])

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Logistic Regression**")

        fig1 = px.imshow(
            cm1,
            text_auto=True,
            x=["Predicted Normal", "Predicted Failure"],
            y=["Actual Normal", "Actual Failure"],
            color_continuous_scale="Blues",
            aspect="auto"
        )

        fig1.update_layout(
            title="Logistic Regression Confusion Matrix"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        st.markdown("**Random Forest**")

        fig2 = px.imshow(
            cm2,
            text_auto=True,
            x=["Predicted Normal", "Predicted Failure"],
            y=["Actual Normal", "Actual Failure"],
            color_continuous_scale="Greens",
            aspect="auto"
        )

        fig2.update_layout(
            title="Random Forest Confusion Matrix"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

    st.subheader("Performance Interpretation")

    st.markdown(
        """
        **Logistic Regression**
        - Higher recall: detects more actual machine failures.
        - Lower precision: produces more false alarms.

        **Random Forest**
        - Higher accuracy and precision.
        - Lower recall: misses more actual machine failures.

        **Important:** The preferred model depends on the operational
        cost of missed failures versus false alarms. Accuracy alone
        is not sufficient for this imbalanced dataset.
        """
    )



elif page == "Failure Prediction":

    st.header("Machine Failure Prediction")

    st.markdown(
        """
        Enter the machine's operating parameters below.
        The selected trained model will predict whether
        the machine is likely to fail.
        """
    )

    st.divider()

    # MODEL SELECTION

    selected_model = st.selectbox(
        "Select Machine Learning Model",
        ["Random Forest", "Logistic Regression"]
    )

    st.divider()

    # MACHINE INPUT PARAMETERS

    st.subheader("Machine Operating Parameters")

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            min_value=295.0,
            max_value=305.0,
            value=300.0,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature (K)",
            min_value=305.0,
            max_value=314.0,
            value=310.0,
            step=0.1
        )

    with col2:

        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=10
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=3.0,
            max_value=77.0,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear (minutes)",
            min_value=0,
            max_value=253,
            value=100,
            step=1
        )

    st.divider()

    # PREDICTION BUTTON

    if st.button("Predict Machine Failure", type="primary"):

        input_data = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature": air_temperature,
            "Process temperature": process_temperature,
            "Rotational speed": rotational_speed,
            "Torque": torque,
            "Tool wear": tool_wear
        }])

        if selected_model == "Logistic Regression":
            model = logistic_model
        else:
            model = random_forest_model

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][
            list(model.classes_).index(1)
        ]

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("⚠️ Machine Failure Predicted")
            else:
                st.success("✅ No Machine Failure Predicted")

        with col2:

            st.metric(
                "Estimated Failure Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(float(probability))

        st.caption(
            "The probability is a model estimate, not a calibrated "
            "guarantee of real-world failure risk."
        )

        with st.expander("View Submitted Machine Parameters"):

            st.dataframe(
                input_data,
                use_container_width=True
            )




elif page == "Feature Importance":

    st.header("Feature Importance Analysis")

    st.markdown(
        """
        Understand which machine operating parameters contribute
        most to the Random Forest model's predictions.
        """
    )

    st.divider()

    # EXTRACT FEATURE IMPORTANCE

    preprocessor = random_forest_model.named_steps["preprocessor"]
    classifier = random_forest_model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    # CLEAN FEATURE NAMES

    feature_labels = []

    for feature in feature_names:

        clean_name = feature.replace("num__", "").replace("cat__", "")

        if clean_name == "Type_L":
            clean_name = "Machine Type L"

        elif clean_name == "Type_M":
            clean_name = "Machine Type M"

        elif clean_name == "Type_H":
            clean_name = "Machine Type H"

        else:
            clean_name = clean_name.replace("_", " ").title()

        feature_labels.append(clean_name)

    # CREATE DATAFRAME

    importance_df = pd.DataFrame({
        "Feature": feature_labels,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df["Importance (%)"] = (
        importance_df["Importance"] * 100
    )

    # TOP THREE FEATURES

    st.subheader("Most Influential Machine Parameters")

    top_features = importance_df.head(3)

    col1, col2, col3 = st.columns(3)

    for column, (_, row) in zip(
        [col1, col2, col3],
        top_features.iterrows()
    ):

        with column:
            st.metric(
                label=row["Feature"],
                value=f'{row["Importance (%)"]:.2f}%'
            )

    st.divider()

    # FEATURE IMPORTANCE CHART

    st.subheader("Feature Importance Ranking")

    chart_df = importance_df.sort_values(
        by="Importance (%)",
        ascending=True
    )                                                          

    fig = px.bar(
        chart_df,
        x="Importance (%)",
        y="Feature",
        orientation="h",
        text="Importance (%)",
        title="Random Forest Feature Importance"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Importance (%)",
        yaxis_title="Machine Features",
        xaxis_range=[
            0,
            importance_df["Importance (%)"].max() * 1.25
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # DETAILED FEATURE IMPORTANCE TABLE

    st.subheader("Detailed Feature Importance")

    display_df = importance_df[
        ["Feature", "Importance (%)"]
    ].reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True
    )

    # DOWNLOAD CSV

    st.download_button(
        label="Download Feature Importance CSV",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name="feature_importance.csv",
        mime="text/csv"
    )

    st.divider()

    # INTERPRETATION

    st.subheader("Interpretation")

    st.markdown(
        """
        Based on the trained Random Forest model:

        - **Rotational Speed:** One of the most influential
          parameters in predicting machine failure.

        - **Torque:** Strong contribution to the model's
          decision-making process.

        - **Tool Wear:** Represents accumulated tool usage
          and contributes to failure prediction.

        Feature importance indicates how much a feature
        contributes to the Random Forest's split decisions.
        It does not establish causation.
        """
    )
