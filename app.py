import streamlit as st
import pandas as pd
import pickle

# === Custom CSS Styling ===
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Title Styling */
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        margin-bottom: 1rem;
        text-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subtitle */
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    /* Container for form */
    .form-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div > div > div {
        border-radius: 12px !important;
        border: 2px solid #e1e8ed !important;
        transition: all 0.3s ease !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Label styling */
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stSlider > label {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        font-size: 1.1rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3) !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 25px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(45deg, #764ba2, #667eea) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Success message styling */
    .success-box {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        box-shadow: 0 10px 20px rgba(68, 160, 141, 0.3);
        margin-top: 1rem;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Columns styling */
    .row-widget.stHorizontal {
        gap: 1rem;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid #e1e8ed !important;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Hide Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .form-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# === Load Model and Version Info from Pickle ===
with open("bigmart_best_model.pkl", "rb") as f:
    model, sklearn_version = pickle.load(f)

# === Title Section ===
st.markdown('<h1 class="main-title">🛒 BigMart Sales Prediction</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Powered by <strong>scikit-learn v{sklearn_version}</strong> • AI-Driven Sales Forecasting</p>', unsafe_allow_html=True)

# === Form Container ===
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # === Create columns for better layout ===
    col1, col2 = st.columns(2)
    
    with col1:
        Item_Identifier = st.text_input("📦 Item Identifier", "FDA15")
        Item_Weight = st.number_input("⚖️ Item Weight (kg)", min_value=0.0, value=12.5)
        Item_Fat_Content = st.selectbox("🥛 Item Fat Content", ["Low Fat", "Regular"])
        Item_Visibility = st.slider("👁️ Item Visibility", min_value=0.0, max_value=0.3, step=0.01, value=0.1)
        Item_Type = st.selectbox("🍎 Item Type", [
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
            "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", 
            "Health and Hygiene", "Hard Drinks", "Canned", "Breads", 
            "Starchy Foods", "Others", "Seafood"
        ])
        Item_MRP = st.number_input("💰 Item MRP (₹)", min_value=0.0, value=150.0)
    
    with col2:
        Outlet_Identifier = st.selectbox("🏪 Outlet Identifier", [
            "OUT027", "OUT013", "OUT049", "OUT035", "OUT046", 
            "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
        ])
        Outlet_Size = st.selectbox("📏 Outlet Size", ["Small", "Medium", "High"])
        Outlet_Location_Type = st.selectbox("📍 Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        Outlet_Type = st.selectbox("🏬 Outlet Type", [
            "Supermarket Type1", "Supermarket Type2", 
            "Supermarket Type3", "Grocery Store"
        ])
        Outlet_Age = st.slider("📅 Outlet Age (Years)", 0, 40, 15)
    
    # === Predict Button ===
    if st.button("🚀 Predict Sales", key="predict_button"):
        input_df = pd.DataFrame([{
            "Item_Identifier": Item_Identifier,
            "Item_Weight": Item_Weight,
            "Item_Fat_Content": Item_Fat_Content,
            "Item_Visibility": Item_Visibility,
            "Item_Type": Item_Type,
            "Item_MRP": Item_MRP,
            "Outlet_Identifier": Outlet_Identifier,
            "Outlet_Size": Outlet_Size,
            "Outlet_Location_Type": Outlet_Location_Type,
            "Outlet_Type": Outlet_Type,
            "Outlet_Age": Outlet_Age
        }])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display result with custom styling
        st.markdown(f"""
        <div class="success-box">
            📈 Predicted Item Outlet Sales: ₹{prediction:,.2f}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# === Footer ===
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: white; opacity: 0.7;">
    <p>Built with ❤️ using Streamlit • Machine Learning Prediction System</p>
</div>
""", unsafe_allow_html=True)
