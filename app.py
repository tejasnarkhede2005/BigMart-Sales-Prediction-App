import streamlit as st
import pandas as pd
import pickle

# === App Configuration ===
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",  # Changed to wide layout for website theme
    initial_sidebar_state="collapsed"
)

# === Load Model and Version Info from Pickle ===
try:
    with open("bigmart_best_model.pkl", "rb") as f:
        model, sklearn_version = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'bigmart_best_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()

# === Custom CSS for Amazon-like Website Theme ===
st.markdown("""
<style>
    /* --- General Body & Layout --- */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        background-color: #131921; /* Darker background */
        color: #FFFFFF; /* All text to white */
    }

    /* --- Main Content Container --- */
    .block-container {
        padding: 2rem 3rem 3rem 3rem !important;
    }
    
    .content-wrapper {
        background-color: #232F3E; /* Main content box to grey/dark blue */
        padding: 2rem 2.5rem;
        border-radius: 8px;
        border: 1px solid #3a4553;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #FFFFFF; /* Ensure text inside is white */
    }

    /* --- Header --- */
    .app-header {
        background-color: #131921; /* Amazon's dark header */
        padding: 1rem 3rem;
        margin: -2rem -3rem 2rem -3rem; /* Extend to full width */
        color: #FFFFFF;
        text-align: center;
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }

    /* --- Top Navigation Bar --- */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2.5rem;
        background-color: #131921; /* Match header */
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #3a4553;
    }
    div[role="radiogroup"] label {
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        font-weight: 600;
        font-size: 1.1rem;
        color: #a6b3bf;
    }
    div[role="radiogroup"] input[type="radio"] { display: none; }
    div[role="radiogroup"] label:has(input:checked) {
        background-color: #3a4553; /* Active nav item grey */
        color: #FFFFFF; /* Active nav text white */
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    div[role="radiogroup"] label:not(:has(input:checked)):hover {
        background-color: #3a4553;
        color: #FFFFFF;
    }

    /* --- Input Widgets Styling --- */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        font-weight: 600;
        color: #FFFFFF; /* Label text white */
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #5a6b7d;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) inset;
        background-color: #3a4553; /* Input background grey */
        font-size: 1rem;
        color: #FFFFFF; /* Input text white */
    }
    /* Ensure selectbox arrow is visible */
    .stSelectbox svg {
        fill: #FFFFFF !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #FF9900; /* Amazon Orange accent */
        box-shadow: 0 0 0 3px rgba(255, 153, 0, 0.2);
    }

    /* --- Slider Styling --- */
    .stSlider .stThumb {
        background-color: #FF9900; /* Keep orange accent */
    }
    .stSlider .stTrack {
        background-color: #5a6b7d;
    }
    .stSlider .stSliderLabel, .stSlider .stTickBar > div {
        color: #FFFFFF !important; /* Make slider numbers white */
    }
    
    /* --- Button Styling --- */
    .stFormSubmitButton > button {
        border-radius: 12px;
        border: none;
        padding: 0.75rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #111; /* Dark text on bright button for contrast */
        background: #FF9900; /* Solid orange button */
        transition: background 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        width: 100%;
    }
    .stFormSubmitButton > button:hover {
        background: #E77600; /* Darker orange on hover */
    }
    
    /* --- Result & Note Box Styling --- */
    [data-testid="stSuccess"] {
        background-color: #3a4553; /* Success box grey */
        border: 1px solid #4dbd74; /* Green border for success */
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    [data-testid="stSuccess"] strong {
        font-size: 2rem;
        color: #FFFFFF; /* Result text white */
    }
    .note-box {
        background-color: #3a4553; /* Note box grey */
        border: 1px solid #5a6b7d;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1.5rem;
    }
    /* Ensure headers within content wrapper are white */
    .content-wrapper h1, .content-wrapper h2, .content-wrapper h3, .content-wrapper h4 {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown('<div class="app-header"><h1>BigMart Sales Predictor</h1></div>', unsafe_allow_html=True)

# === Top Navigation Bar ===
page = st.radio(
    "Navigation",
    ["Home", "About", "Help"],
    horizontal=True,
    label_visibility="collapsed"
)

# === Page Content Logic ===
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

if "Home" in page:
    st.header("Sales Prediction Form")
    st.markdown("Fill in the details below to generate a sales prediction.")
    st.markdown("---")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("🍔 Item Fat Content", ["Low Fat", "Regular"], key="Item_Fat_Content")
            st.selectbox("📦 Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"], key="Item_Type")
            st.number_input("💵 Item MRP (₹)", min_value=0.0, value=150.0, key="Item_MRP")
            st.slider("👁️ Item Visibility", min_value=0.0, max_value=0.35, step=0.01, value=0.05, key="Item_Visibility")
            st.number_input("⚖️ Item Weight (kg)", min_value=0.0, value=12.5, key="Item_Weight")

        with col2:
            st.selectbox("🏪 Outlet Identifier", ["OUT027", "OUT013", "OUT049", "OUT035", "OUT046", "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"], key="Outlet_Identifier")
            st.selectbox("📏 Outlet Size", ["Small", "Medium", "High"], key="Outlet_Size")
            st.selectbox("🌍 Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"], key="Outlet_Location_Type")
            st.selectbox("🏬 Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"], key="Outlet_Type")
            st.slider("📅 Outlet Age (Years)", 0, 40, 15, key="Outlet_Age")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Predict Sales")

    if submit:
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        input_df = pd.DataFrame([{"Item_Identifier": "FAD123", "Item_Weight": st.session_state.Item_Weight, "Item_Fat_Content": st.session_state.Item_Fat_Content, "Item_Visibility": st.session_state.Item_Visibility, "Item_Type": st.session_state.Item_Type, "Item_MRP": st.session_state.Item_MRP, "Outlet_Identifier": st.session_state.Outlet_Identifier, "Outlet_Size": st.session_state.Outlet_Size, "Outlet_Location_Type": st.session_state.Outlet_Location_Type, "Outlet_Type": st.session_state.Outlet_Type, "Outlet_Age": st.session_state.Outlet_Age}])
        with st.spinner('Predicting...'):
            try:
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted Sales: **₹{prediction:,.2f}**")
                st.markdown("""<div class="note-box"><b>Note:</b> This is an estimate based on historical data. Actual sales can vary due to promotions, seasonality, etc.</div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif "About" in page:
    st.header("About This Project")
    st.markdown(f"""
    This application provides a sales prediction for the BigMart dataset using a machine learning model. It helps managers forecast sales to optimize inventory and marketing strategies.
    
    #### How It Works
    The prediction is generated by a pre-trained regression model that considers various product and store features:
    - **Product Features:** Item weight, fat content, visibility, type, and MRP.
    - **Store Features:** Outlet size, location type, store type, and age.
    
    This tool showcases how data science can be applied to solve real-world retail challenges.
    
    **Model Version:** scikit-learn v{sklearn_version} | **Developer:** Tejas
    """, unsafe_allow_html=True)

elif "Help" in page:
    st.header("Help & Instructions")
    st.markdown("""
    #### How to Use
    1.  **Fill the Form:** On the **Home** screen, enter all the details for the product and the store in the two columns.
    2.  **Adjust Sliders:** Use the sliders for `Item Visibility` and `Outlet Age`.
    3.  **Predict:** Click the **"Predict Sales"** button at the bottom of the form.
    4.  **View Result:** The estimated sales amount will appear below the form.

    #### FAQ
    **Q: How accurate is the prediction?**
    
    A: The prediction is a data-driven estimate. Real-world sales can be influenced by many factors not included in the model, like holidays or special promotions.

    **Q: Why is "Item Identifier" not an input?**
    
    A: The model was trained in a way that it doesn't require the specific item ID for prediction, as it learns from the other features provided.
    """)

st.markdown('</div>', unsafe_allow_html=True)

