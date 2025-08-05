import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ============ Load model, features, and average duration ============
model = joblib.load("model.pkl")
features_used = joblib.load("features_used.pkl")
avg_duration = joblib.load("avg_duration.pkl")  # New

# ============ Page Config ============
st.set_page_config(page_title="NYC Taxi Duration Predictor", layout="centered")
st.title("🚖 NYC Taxi Trip Duration Prediction")
st.markdown(
    "Predict how long a taxi ride in NYC will take using a trained ML model.")
st.markdown("---")

# ============ User Inputs ============
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        distance_km = st.slider("Distance (km)", 0.1,
                                50.0, step=0.1, value=3.5)
        pickup_hour = st.slider("Pickup Hour (0–23)", 0, 23, value=14)
        pickup_weekday = st.selectbox("Pickup Weekday", [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    with col2:
        pickup_month = st.selectbox(
            "Pickup Month", list(range(1, 13)), index=5)
        store_and_fwd_flag = st.radio("Store & Forward Flag", ["N", "Y"])

    submitted = st.form_submit_button("🚕 Predict")

# ============ Helper Functions ============


def get_time_of_day(hour):
    if 0 <= hour <= 5:
        return 'night'
    elif 6 <= hour <= 11:
        return 'morning'
    elif 12 <= hour <= 17:
        return 'afternoon'
    else:
        return 'evening'


def get_distance_bin(km):
    if km <= 2:
        return 'short'
    elif km <= 5:
        return 'medium'
    elif km <= 10:
        return 'long'
    else:
        return 'very_long'


# ============ Prediction ============
if submitted:
    # Feature Engineering
    weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                   "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    pickup_weekday_num = weekday_map[pickup_weekday]
    is_weekend = 1 if pickup_weekday_num in [5, 6] else 0
    time_of_day = get_time_of_day(pickup_hour)
    distance_bin = get_distance_bin(distance_km)

    # Input dictionary
    input_dict = {
        "distance_km": distance_km,
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday_num,
        "pickup_month": pickup_month,
        "store_and_fwd_flag": 1 if store_and_fwd_flag == "Y" else 0,
        "is_weekend": is_weekend
    }

    for tod in ['morning', 'afternoon', 'evening', 'night']:
        input_dict[f"time_of_day_{tod}"] = 1 if time_of_day == tod else 0

    for bin_name in ['short', 'medium', 'long', 'very_long']:
        input_dict[f"distance_bin_{bin_name}"] = 1 if distance_bin == bin_name else 0

    input_df = pd.DataFrame([input_dict])

    for col in features_used:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features_used]

    # Predict
    log_pred = model.predict(input_df)[0]
    duration_sec = np.expm1(log_pred)
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)

    # Display Result
    st.markdown("### 🧾 Prediction Result:")
    st.success(
        f"🕒 **Estimated Trip Duration:** {minutes} minutes, {seconds} seconds")
    st.caption(f"(= {round(duration_sec)} seconds)")

    # 📊 Visual Comparison with NYC average
    st.markdown("---")
    st.markdown("### 📊 Trip Duration: Predicted vs NYC Average")

    fig, ax = plt.subplots()
    bars = ax.bar(["Predicted", "NYC Average"], [
                  duration_sec, avg_duration], color=["teal", "gray"])
    ax.set_ylabel("Duration (seconds)")
    ax.set_ylim(0, max(duration_sec, avg_duration) + 300)
    ax.set_title("Trip Duration Comparison")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 20,
                f"{int(yval)}s", ha='center', fontsize=10)

    st.pyplot(fig)

    # Display Inputs
    st.markdown("---")
    st.markdown("### 📋 Your Input Summary:")
    st.json(input_dict)
