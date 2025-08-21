
import streamlit as st
import requests
import time
from streamlit_lottie import st_lottie

# Load animation from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Set Streamlit page configuration
st.set_page_config(page_title="Smart Health Cluster", page_icon="🧠", layout="centered")

# Load a Lottie animation
lottie_health = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_q5pk6p1k.json")

if lottie_health:
    st_lottie(lottie_health, height=300, key="health")
else:
    st.warning("Animation couldn't be loaded.")

# Sidebar
with st.sidebar:
    st.title("🩺 About")
    st.markdown("""
    This is an AI-powered **Community Health Dashboard** that uses public health indicators to predict a community’s health cluster.
    
    Built with 💡 and care.
    """)
    st.markdown("Made with ❤️ using Streamlit")

# Title with gradient text
st.markdown("""
    <h1 style='text-align: center; color: #2E8B57; font-size: 48px; background: linear-gradient(to right, #43cea2, #185a9d); -webkit-background-clip: text; color: transparent;'>Smart Community Health Predictor</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# Input form
with st.form("health_form"):
    st.subheader("📊 Enter Community Health Data")

    col1, col2 = st.columns(2)

    with col1:
        sex_ratio = st.number_input("Sex Ratio (Females per 1000 Males)", 500, 1200, 950)
        anaemic_15_19 = st.slider("Anaemic Women 15-19 (%)", 0, 100, 40)
        anaemic_15_49 = st.slider("Anaemic Women 15-49 (%)", 0, 100, 50)
        delivery_cost = st.number_input("Average Delivery Cost (₹)", 0, 50000, 2000)

    with col2:
        skilled_births = st.slider("Skilled Birth Attendance (%)", 0, 100, 75)
        immunization = st.slider("Fully Immunized Children (%)", 0, 100, 80)
        household_size = st.number_input("Mean Household Size", 1, 15, 5)
        urban_population = st.slider("Urban Population (%)", 0, 100, 45)

    submit = st.form_submit_button("🔍 Predict Health Cluster")

# Handle submit
if submit:
    with st.spinner("Predicting..."):
        time.sleep(1.5)  # Simulated delay

        # Simulated logic for prediction
        cluster = "Healthy" if immunization > 70 and anaemic_15_49 < 55 else "At Risk"

    st.success(f"🏥 Predicted Health Cluster: **{cluster}**")

    if cluster == "Healthy":
        st.balloons()
        st.markdown("### ✅ Your community shows strong health indicators!")
    else:
        st.markdown("### ⚠️ Your community may need health interventions.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2025 Smart Health AI</p>", unsafe_allow_html=True)
