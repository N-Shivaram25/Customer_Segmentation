import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown('''
<style>
.block-container {padding-top: 2rem; max-width: 1450px;}
.hero {
    padding: 1.5rem 1.8rem; border-radius: 18px; margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #172554, #1e3a8a);
}
.hero h1 {color: white; margin: 0 0 .3rem 0;}
.hero p {color: #dbeafe; margin: 0;}
.segment-card {
    padding: 1.4rem 1.6rem; border-radius: 16px;
    background: #111827; border: 1px solid #334155; margin-bottom: 1rem;
}
.segment-card h2 {color: #f8fafc; margin: 0 0 .4rem 0;}
.segment-card p {color: #cbd5e1; margin: .25rem 0;}
.strategy-card {
    padding: 1.2rem 1.5rem; border-radius: 16px;
    background: #0f172a; border-left: 5px solid #38bdf8; margin-bottom: 1rem;
}
.strategy-card h3 {color: #f8fafc; margin-top: 0;}
.strategy-card p {color: #cbd5e1; margin: 0;}
.section-title {font-size: 1.45rem; font-weight: 700; margin-top: 1.8rem;}
.section-subtitle {color: #94a3b8; margin-bottom: 1rem;}
div[data-testid="stMetric"] {
    background: #111827; border: 1px solid #334155;
    padding: 1rem; border-radius: 14px;
}
</style>
''', unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    return scaler, kmeans

try:
    scaler, kmeans = load_models()
except Exception as e:
    st.error("Could not load scaler.pkl or kmeans_model.pkl.")
    st.code(str(e))
    st.stop()

# ---------- Cluster information ----------
cluster_profile = pd.DataFrame({
    "Income": [74349.69, 35337.09, 54234.96],
    "Recency": [49.05, 48.75, 50.02],
    "Tenure_Days": [334.40, 306.12, 493.17],
    "Total_Spending": [1235.78, 108.18, 699.31],
    "Total_Purchases": [19.13, 6.12, 16.26],
    "NumDealsPurchases": [1.32, 1.86, 5.04],
    "NumWebVisitsMonth": [3.06, 6.35, 6.68],
    "Total_Children": [0.35, 1.20, 1.38]
}, index=[0, 1, 2])

cluster_counts = {0: 750, 1: 1035, 2: 455}

segment_names = {
    0: "High-Value Customers",
    1: "Low-Value / Low-Conversion Customers",
    2: "Deal-Oriented & Digitally Active Customers"
}

segment_icons = {0: "💎", 1: "📉", 2: "🎯"}

recommendations = {
    0: "Focus on retention, loyalty rewards, premium offers, and cross-selling.",
    1: "Focus on personalized recommendations, targeted offers, and conversion campaigns.",
    2: "Focus on promotional campaigns, discounts, online offers, and re-engagement."
}

# ---------- Header ----------
st.markdown('''
<div class="hero">
<h1>👥 Customer Segmentation Dashboard</h1>
<p>Use the trained K-Means model to identify customer segments and understand the business strategy for each segment.</p>
</div>
''', unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("Customer Information")
st.sidebar.caption("Enter customer behavioral and demographic information.")

st.sidebar.subheader("💰 Financial Information")
income = st.sidebar.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
total_spending = st.sidebar.number_input("Total Spending", min_value=0.0, value=600.0, step=50.0)

st.sidebar.subheader("🛒 Purchase Behaviour")
total_purchases = st.sidebar.number_input("Total Purchases", min_value=0, value=12, step=1)
deals_purchases = st.sidebar.number_input("Deal Purchases", min_value=0, value=2, step=1)
web_visits = st.sidebar.number_input("Web Visits per Month", min_value=0, value=5, step=1)

st.sidebar.subheader("👤 Customer Profile")
recency = st.sidebar.number_input("Recency", min_value=0, value=50, step=1)
tenure_days = st.sidebar.number_input("Tenure (Days)", min_value=0, value=500, step=10)
total_children = st.sidebar.number_input("Total Children", min_value=0, value=1, step=1)

predict = st.sidebar.button("🎯 Predict Customer Segment", use_container_width=True, type="primary")

input_data = pd.DataFrame({
    "Income": [income],
    "Recency": [recency],
    "Tenure_Days": [tenure_days],
    "Total_Spending": [total_spending],
    "Total_Purchases": [total_purchases],
    "NumDealsPurchases": [deals_purchases],
    "NumWebVisitsMonth": [web_visits],
    "Total_Children": [total_children]
})

# ---------- Prediction ----------
if predict:
    try:
        input_scaled = scaler.transform(input_data)
        cluster = int(kmeans.predict(input_scaled)[0])
    except Exception as e:
        st.error("Prediction failed. Make sure the scaler and K-Means model use the same 8 features.")
        st.code(str(e))
        st.stop()

    segment = segment_names.get(cluster, "Customer Segment")
    icon = segment_icons.get(cluster, "👤")
    strategy = recommendations.get(cluster, "Develop a targeted customer strategy.")

    st.markdown('<div class="section-title">🎯 Customer Segment</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="segment-card">
        <h2>{icon} {segment}</h2>
        <p><strong>Cluster ID:</strong> {cluster}</p>
        <p>This customer has been assigned to K-Means cluster {cluster} based on the selected features.</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section-title">💼 Recommended Business Strategy</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="strategy-card">
        <h3>Recommended Action</h3>
        <p>{strategy}</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📌 Customer Snapshot</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Income", f"₹{income:,.0f}")
    b.metric("Total Spending", f"₹{total_spending:,.0f}")
    c.metric("Total Purchases", f"{total_purchases}")
    d.metric("Web Visits / Month", f"{web_visits}")

    # Standardized comparison prevents Income from dominating the visualization.
    st.markdown('<div class="section-title">📊 Customer vs Segment Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Standardized values make features comparable on the same scale.</div>',
        unsafe_allow_html=True
    )

    combined = pd.concat([cluster_profile, input_data], ignore_index=True)
    means = combined.mean()
    stds = combined.std(ddof=0).replace(0, 1)

    profile_scaled = (cluster_profile - means) / stds
    customer_scaled = (input_data.iloc[0] - means) / stds

    comparison = pd.DataFrame({
        "Customer": customer_scaled,
        "Segment Average": profile_scaled.loc[cluster]
    })

    st.bar_chart(comparison, height=430)

    with st.expander("🔎 View Customer Input"):
        st.dataframe(input_data, use_container_width=True, hide_index=True)

# ---------- Distribution ----------
st.markdown('<div class="section-title">📈 Customer Segment Distribution</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Distribution of customers across the three K-Means segments.</div>',
    unsafe_allow_html=True
)

distribution = pd.DataFrame({
    "Segment": ["High-Value", "Low-Value / Low-Conversion", "Deal-Oriented / Digital"],
    "Customers": [750, 1035, 455]
}).set_index("Segment")

left, right = st.columns([2.3, 1])
with left:
    st.bar_chart(distribution, height=350)
with right:
    st.metric("High-Value", "750 customers")
    st.metric("Low-Value / Low-Conversion", "1,035 customers")
    st.metric("Deal-Oriented / Digital", "455 customers")

# ---------- Profile table ----------
st.markdown('<div class="section-title">📋 Segment Profile Comparison</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Average customer characteristics identified for each K-Means segment.</div>',
    unsafe_allow_html=True
)

display_profile = cluster_profile.copy()
display_profile.index = ["High-Value", "Low-Value / Low-Conversion", "Deal-Oriented / Digital"]

st.dataframe(display_profile.round(2), use_container_width=True)

st.caption(
    "K-Means model • Features: Income, Recency, Tenure, Spending, Purchases, "
    "Deal Purchases, Web Visits and Children"
)
