import streamlit as st
import pandas as pd
import pickle

# === Load Model and Version Info from Pickle ===
try:
    with open("bigmart_best_model.pkl", "rb") as f:
        model, sklearn_version = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'bigmart_best_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()

# === App Configuration ===
st.set_page_config(
    page_title="BigMart Sales Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Custom CSS for Professional Styling ===
st.markdown("""
<style>
    /* General Styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f0f2f6;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 5rem;
        padding-bottom: 5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Navbar */
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 999;
    }
    .navbar-brand {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1e3a8a; /* A deep blue color */
    }

    /* Title and Header Styles */
    h1, h2, h3 {
        color: #1e3a8a;
    }
    
    /* Card/Container for the form */
    .form-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }

    /* Input Widgets Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        padding: 0.5rem 0;
    }

    /* Button Styling */
    .stButton > button, .stFormSubmitButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        background-color: #2563eb; /* A nice blue */
        transition: background-color 0.3s, box-shadow 0.3s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #1d4ed8; /* Darker blue on hover */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Success Box Styling */
    [data-testid="stSuccess"] {
        background-color: #e0f2fe;
        border-left: 5px solid #0ea5e9;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Note Box Styling */
    .note-box {
        background-color: #fefce8; /* Light yellow */
        border: 1px solid #facc15; /* Yellow border */
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #713f12; /* Dark yellow text */
    }
    
    /* Expander Styling */
    .stExpander {
        border: 1px solid #d1d5db;
        border-radius: 10px;
        background-color: #fafafa;
    }
    
    /* Footer */
    footer {
        text-align: center;
        padding: 1.5rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)


# === Professional Navbar ===
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">🛒 BigMart Sales Predictor</div>
</div>
""", unsafe_allow_html=True)

# === Title & Introduction ===
st.title("BigMart Sales Prediction App")
st.markdown(f"""
Welcome! This tool uses a machine learning model (**scikit-learn v{sklearn_version}**) to predict item sales.
Fill in the details below to get an estimate.
""")

# === Create Form for Inputs ===
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.header("🧾 Enter Item & Outlet Details")
        col1, col2 = st.columns(2)

        with col1:
            Item_Identifier = st.text_input("🆔 Item Identifier", "FDA15")
            Item_Weight = st.number_input("⚖️ Item Weight (kg)", min_value=0.0, value=12.5)
            Item_Fat_Content = st.selectbox("🍔 Item Fat Content", ["Low Fat", "Regular"])
            Item_Visibility = st.slider("👁️ Item Visibility", min_value=0.0, max_value=0.3, step=0.01, value=0.1)
            Item_Type = st.selectbox("📦 Item Type", [
                "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
                "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
                "Health and Hygiene", "Hard Drinks", "Canned", "Breads",
                "Starchy Foods", "Others", "Seafood"
            ])
            Item_MRP = st.number_input("💵 Item MRP (₹)", min_value=0.0, value=150.0)

        with col2:
            Outlet_Identifier = st.selectbox("🏪 Outlet Identifier", [
                "OUT027", "OUT013", "OUT049", "OUT035", "OUT046",
                "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
            ])
            Outlet_Size = st.selectbox("📏 Outlet Size", ["Small", "Medium", "High"])
            Outlet_Location_Type = st.selectbox("🌍 Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
            Outlet_Type = st.selectbox("🏬 Outlet Type", [
                "Supermarket Type1", "Supermarket Type2",
                "Supermarket Type3", "Grocery Store"
            ])
            Outlet_Age = st.slider("📅 Outlet Age (Years)", 0, 40, 15)

        # === Submit Button ===
        submit = st.form_submit_button("🔍 Predict Sales")

    st.markdown('</div>', unsafe_allow_html=True)

# === On Submit: Make Prediction ===
if submit:
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
    
    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"📈 **Predicted Item Outlet Sales:** ₹{prediction:,.2f}")

        st.markdown("""
        <div class="note-box">
            <b>🧠 Note:</b> This prediction is based on historical sales data. Actual sales may vary due to external factors like promotions, seasonality, etc.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ **Prediction failed:** An error occurred. Please check your inputs. Details: {e}")

# === Expandable Info Section ===
with st.expander("📘 About this Model"):
    st.markdown("""
    This sales prediction model was trained on the **BigMart** dataset.
    The model leverages several techniques to ensure accuracy:
    - **Feature Engineering:** Converting establishment year to outlet age.
    - **Data Preprocessing:** One-hot encoding for categorical variables and normalization for numerical features.
    - **Hyperparameter Tuning:** Optimizing the model's parameters for the best performance.

    - **Target Variable:** `Item_Outlet_Sales` (₹)
    - **Model Type:** Regression
    - **Use Case:** Forecasting sales to optimize stock and marketing strategies.
    """)

# === Footer ===
st.markdown("---")
st.markdown("<footer>Developed with ❤️ using Streamlit | Built by Tejas</footer>", unsafe_allow_html=True)
