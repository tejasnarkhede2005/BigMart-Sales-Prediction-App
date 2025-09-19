import streamlit as st
import pandas as pd
import pickle

# === Fashionable CSS Styling ===
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');

    body {
        font-family: 'Poppins', sans-serif !important;
        background: radial-gradient(circle at top left, #141E30, #243B55);
        color: #ffffff;
    }

    /* Animated Gradient Title */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.2rem;
        text-align: center;
        font-weight: 700;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #00F260, #0575E6, #ff6b6b, #f9d423);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: flowing 6s ease infinite;
        margin-bottom: 1rem;
        text-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }

    @keyframes flowing {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #e0e0e0;
        margin-bottom: 2rem;
    }

    /* Form container (glassmorphism) */
    .form-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 35px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .form-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.55);
    }

    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stSelectbox div, .stSlider {
        border-radius: 10px !important;
        border: 1.5px solid #444 !important;
        padding: 0.4rem 0.6rem !important;
        background: rgba(255,255,255,0.1) !important;
        color: #fff !important;
    }

    /* Label Styling */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #ffcc70 !important;
        font-weight: 600 !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff6a00, #ee0979) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 18px rgba(238,9,121,0.4) !important;
    }
    .stButton > button:hover {
        transform: scale(1.05) !important;
        background: linear-gradient(90deg, #ee0979, #ff6a00) !important;
        box-shadow: 0 12px 24px rgba(238,9,121,0.6) !important;
    }

    /* Prediction Result */
    .success-box {
        margin-top: 2rem;
        padding: 1.5rem;
        text-align: center;
        border-radius: 14px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        box-shadow: 0 12px 25px rgba(0, 114, 255, 0.4);
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #bbb;
        margin-top: 2rem;
        font-size: 0.9rem;
    }

    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
