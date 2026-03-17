import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============ Page Config & Rebranding ============
st.set_page_config(
    page_title="MetroRoute AI Pro",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium SaaS Look
st.markdown("""
    <style>
    /* Hide default header */
    header[data-testid="stHeader"] { visibility: hidden; }
    
    /* Main Background & Font */
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Custom Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00FFA2;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #A1A1AA;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1A1C24;
        border-right: 1px solid #2D2E35;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #00FFA2;
        color: #0E1117;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00CC81;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 162, 0.3);
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
        background-color: #1A1C24;
    }
    </style>
    """, unsafe_allow_html=True)

# ============ Load Assets ============
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.pkl")
        features_used = joblib.load("features_used.pkl")
        avg_duration = joblib.load("avg_duration.pkl")
        return model, features_used, avg_duration
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

model, features_used, avg_duration_sec = load_assets()

# ============ Sidebar: Input Form ============
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1048/1048313.png", width=80)
    st.title("MetroRoute AI")
    st.markdown("---")
    
    with st.form("trip_input_form"):
        st.subheader("📍 Journey Details")
        distance_km = st.slider("Trip Distance (km)", 0.1, 50.0, 3.5, step=0.1)
        
        st.subheader("⏰ Timing")
        pickup_hour = st.select_slider("Pickup Hour", options=list(range(24)), value=14)
        pickup_weekday = st.selectbox("Day of Week", [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])
        pickup_month = st.selectbox("Month", options=list(range(1, 13)), index=5)
        
        st.subheader("⚙️ Meta")
        store_and_fwd_flag = st.radio("S&F Flag", ["N", "Y"], horizontal=True)
        
        submitted = st.form_submit_button("⚡ Calculate Prediction")

# ============ Helper Functions ============
def get_time_of_day(hour):
    if 0 <= hour <= 5: return 'night'
    elif 6 <= hour <= 11: return 'morning'
    elif 12 <= hour <= 17: return 'afternoon'
    else: return 'evening'

def get_distance_bin(km):
    if km <= 2: return 'short'
    elif km <= 5: return 'medium'
    elif km <= 10: return 'long'
    else: return 'very_long'

# ============ Main App Content ============
# Tabs for better organization
tab_dashboard, tab_insights, tab_about = st.tabs([
    "🏠 Predictive Analytics", 
    "📊 Feature Insights", 
    "ℹ️ Project Blueprint"
])

# Process Prediction if clicked
if submitted:
    # Feature Engineering
    weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    pickup_weekday_num = weekday_map[pickup_weekday]
    is_weekend = 1 if pickup_weekday_num in [5, 6] else 0
    time_of_day = get_time_of_day(pickup_hour)
    distance_bin = get_distance_bin(distance_km)

    input_dict = {
        "distance_km": distance_km, "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday_num, "pickup_month": pickup_month,
        "store_and_fwd_flag": 1 if store_and_fwd_flag == "Y" else 0,
        "is_weekend": is_weekend
    }
    for tod in ['morning', 'afternoon', 'evening', 'night']:
        input_dict[f"time_of_day_{tod}"] = 1 if time_of_day == tod else 0
    for bin_name in ['short', 'medium', 'long', 'very_long']:
        input_dict[f"distance_bin_{bin_name}"] = 1 if distance_bin == bin_name else 0

    input_df = pd.DataFrame([input_dict])
    for col in features_used:
        if col not in input_df.columns: input_df[col] = 0
    input_df = input_df[features_used]

    # Predict
    log_pred = model.predict(input_df)[0]
    duration_sec = np.expm1(log_pred)
    st.session_state['pred_results'] = {
        'sec': duration_sec,
        'min': int(duration_sec // 60),
        'rem_sec': int(duration_sec % 60),
        'input_dict': input_dict
    }

# --- Tab 1: Dashboard ---
with tab_dashboard:
    if 'pred_results' in st.session_state:
        res = st.session_state['pred_results']
        
        st.title("🚀 Prediction Intelligence")
        
        # Metric Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Journey", f"{res['min']} min {res['rem_sec']} s")
        m2.metric("Total Seconds", f"{int(res['sec'])}s")
        diff = res['sec'] - avg_duration_sec
        m3.metric("Vs NYC Average", f"{int(abs(diff))}s", delta=f"{int(diff)}s", delta_color="inverse")
        
        st.markdown("---")
        
        # Visuals Section
        col_left, col_right = st.columns([1.2, 1])
        
        with col_left:
            st.subheader("📈 Performance Benchmark")
            df_plot = pd.DataFrame({
                "Metric": ["Predicted", "NYC Average"],
                "Seconds": [res['sec'], avg_duration_sec]
            })
            fig_bar = px.bar(
                df_plot, x="Metric", y="Seconds", color="Metric",
                color_discrete_map={"Predicted": "#00FFA2", "NYC Average": "#31333F"},
                text_auto='.0f'
            )
            fig_bar.update_layout(showlegend=False, height=400, template="plotly_dark", margin=dict(t=20, b=20, l=0, r=0))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_right:
            st.subheader("🎯 Congestion Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res['sec'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Trip Duration (sec)", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, max(res['sec'], avg_duration_sec) * 1.5], 'tickwidth': 1},
                    'bar': {'color': "#00FFA2"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, avg_duration_sec], 'color': 'rgba(0, 255, 162, 0.1)'},
                        {'range': [avg_duration_sec, max(res['sec'], avg_duration_sec) * 1.5], 'color': 'rgba(255, 75, 75, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_duration_sec
                    }
                }
            ))
            fig_gauge.update_layout(height=400, template="plotly_dark", margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Dynamic insight boxes
        if diff > 0:
            st.warning(f"⚠️ **Traffic Alert**: This route is predicted to take **{int(diff)} seconds longer** than the city average. Heavy traffic or peak hours likely.")
        else:
            st.success(f"✅ **Efficient Route**: You are beating the average by **{int(abs(diff))} seconds**! Expect a smooth ride.")
            
    else:
        # Welcome Screen
        st.title("🏙️ Welcome to MetroRoute AI Pro")
        st.info("Your intelligent companion for NYC trip duration analytics. Configure your journey in the sidebar to begin.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🗺️ Efficient Planning")
            st.write("Leverage machine learning to predict trip times based on historical NYC taxi data patterns.")
        with c2:
            st.markdown("### 📊 Interactive Visuals")
            st.write("Compare predictions against citywide averages with dynamic charts and congestion gauges.")
        with c3:
            st.markdown("### ⚡ Real-time Inference")
            st.write("Instant predictions powered by optimized regression models and feature engineering.")

# --- Tab 2: Feature Insights ---
with tab_insights:
    st.title("📊 Data & Feature Architecture")
    st.write("Understand the engine behind the predictions. These are the key variables influencing your trip duration.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🏗️ Feature Breakdown
        * **Distance (km):** The primary driver of duration.
        * **Pickup Hour:** Captures peak traffic vs. midnight flow.
        * **Weekday:** Weekend vs. Workday speed variations.
        * **Time of Day Bin:** Categorical grouping for neural/tree model sensitivity.
        * **Distance Bin:** Categorizes trips as short, medium, or long-haul.
        """)
    
    with col2:
        if 'pred_results' in st.session_state:
            st.markdown("#### 🔍 Active Input Vector")
            st.json(st.session_state['pred_results']['input_dict'])
        else:
            st.info("Submit a prediction to see the input vector details here.")

# --- Tab 3: About Project ---
with tab_about:
    st.title("ℹ️ Project Blueprint")
    st.markdown("""
    ### 🚕 NYC Taxi Trip Duration Prediction
    This project aims to build a robust machine learning pipeline to predict the total ride duration of NYC taxi trips.
    
    #### 🛠️ Tech Stack
    - **Modeling:** Python, Scikit-Learn, XGBoost/Linear Regression.
    - **UI/UX:** Streamlit, Plotly, Custom CSS.
    - **Deployment:** Production-grade SaaS layout.
    
    #### 📂 Dataset Context
    The model is trained on the **NYC Taxi & Limousine Commission (TLC)** dataset, featuring millions of trip records. It accounts for geographical distance, temporal patterns (hour/month), and traffic fluctuations based on historical data.
    
    ---
    *Developed as a high-fidelity logistics dashboard showcase.*
    """)
