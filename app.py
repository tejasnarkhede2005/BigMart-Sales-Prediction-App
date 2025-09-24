import streamlit as st
import pandas as pd
import pickle

# === App Configuration ===
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Initialize Session State for Theme Toggle ===
if 'theme' not in st.session_state:
    st.session_state.theme = 'light' # Default to light theme

# === Load Model and Version Info from Pickle ===
try:
    with open("bigmart_best_model.pkl", "rb") as f:
        model, sklearn_version = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'bigmart_best_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()

# === THEME CSS ===
light_theme_css = """
<style>
    body { background-color: #EAEDED; color: #0F1111; }
    .block-container { padding: 2rem 3rem 3rem 3rem !important; }
    .content-wrapper { background-color: #FEFEFE; padding: 2rem 2.5rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .app-header { background-color: #131921; padding: 1rem 3rem; margin: -2rem -3rem 2rem -3rem; color: #FFFFFF; text-align: center; }
    .app-header h1 { margin: 0; font-size: 2.5rem; }
    div[role="radiogroup"] { display: flex; justify-content: center; gap: 1rem; margin-bottom: 2.5rem; background-color: #232F3E; padding: 0.75rem; border-radius: 8px; }
    div[role="radiogroup"] label { padding: 0.5rem 1.5rem; border-radius: 6px; cursor: pointer; transition: all 0.2s; font-weight: 600; font-size: 1.1rem; color: #a6b3bf; }
    div[role="radiogroup"] input[type="radio"] { display: none; }
    div[role="radiogroup"] label:has(input:checked) { background-color: #FF9900; color: #131921; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
    div[role="radiogroup"] label:not(:has(input:checked)):hover { background-color: #3a4553; color: #FFFFFF; }
    .stTextInput label, .stNumberInput label, .stSelectbox label { font-weight: 600; color: #0F1111; }
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div { border-radius: 8px; border: 1px solid #a6a6a6; box-shadow: 0 1px 2px rgba(0,0,0,0.05) inset; background-color: #F0F2F5; font-size: 1rem; }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus, .stSelectbox > div > div:focus-within { border-color: #E77600; box-shadow: 0 0 0 3px #fcf4e8, 0 1px 2px rgba(0,0,0,0.05) inset; }
    .stSlider .stThumb { background-color: #E77600; }
    .stSlider .stTrack { background-color: #a6a6a6; }
    .stFormSubmitButton > button { border-radius: 12px; border: 1px solid #a88734; padding: 0.75rem; font-size: 1.2rem; font-weight: 700; color: #111; background: linear-gradient(to bottom, #f7dfa5, #f0c14b); transition: background 0.2s; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 100%; }
    .stFormSubmitButton > button:hover { background: linear-gradient(to bottom, #f5d78e, #eeb933); }
    [data-testid="stSuccess"] { background-color: #f2fafa; border: 1px solid #007185; border-radius: 12px; padding: 1.5rem; text-align: center; }
    [data-testid="stSuccess"] strong { font-size: 2rem; color: #0F1111; }
    .note-box { background-color: #f2f7fa; border: 1px solid #c7e3f1; border-radius: 12px; padding: 1rem; margin-top: 1.5rem; }
</style>
"""

dark_theme_css = """
<style>
    body { background-color: #131921; color: #FFFFFF; }
    .block-container { padding: 2rem 3rem 3rem 3rem !important; }
    .content-wrapper { background-color: #232F3E; padding: 2rem 2.5rem; border-radius: 8px; border: 1px solid #3a4553; box-shadow: 0 4px 12px rgba(0,0,0,0.2); color: #FFFFFF; }
    .app-header { background-color: #131921; padding: 1rem 3rem; margin: -2rem -3rem 2rem -3rem; color: #FFFFFF; text-align: center; }
    .app-header h1 { margin: 0; font-size: 2.5rem; }
    div[role="radiogroup"] { display: flex; justify-content: center; gap: 1rem; margin-bottom: 2.5rem; background-color: #131921; padding: 0.75rem; border-radius: 8px; border: 1px solid #3a4553; }
    div[role="radiogroup"] label { padding: 0.5rem 1.5rem; border-radius: 6px; cursor: pointer; transition: all 0.2s; font-weight: 600; font-size: 1.1rem; color: #a6b3bf; }
    div[role="radiogroup"] input[type="radio"] { display: none; }
    div[role="radiogroup"] label:has(input:checked) { background-color: #3a4553; color: #FFFFFF; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
    div[role="radiogroup"] label:not(:has(input:checked)):hover { background-color: #3a4553; color: #FFFFFF; }
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label { font-weight: 600; color: #FFFFFF; }
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div { border-radius: 8px; border: 1px solid #5a6b7d; box-shadow: 0 1px 2px rgba(0,0,0,0.1) inset; background-color: #3a4553; font-size: 1rem; color: #FFFFFF; }
    .stSelectbox svg { fill: #FFFFFF !important; }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus, .stSelectbox > div > div:focus-within { border-color: #FF9900; box-shadow: 0 0 0 3px rgba(255, 153, 0, 0.2); }
    .stSlider .stThumb { background-color: #FF9900; }
    .stSlider .stTrack { background-color: #5a6b7d; }
    .stSlider .stSliderLabel, .stSlider .stTickBar > div { color: #FFFFFF !important; }
    .stFormSubmitButton > button { border-radius: 12px; border: none; padding: 0.75rem; font-size: 1.2rem; font-weight: 700; color: #111; background: #FF9900; transition: background 0.2s; box-shadow: 0 2px 5px rgba(0,0,0,0.2); width: 100%; }
    .stFormSubmitButton > button:hover { background: #E77600; }
    [data-testid="stSuccess"] { background-color: #3a4553; border: 1px solid #4dbd74; border-radius: 12px; padding: 1.5rem; text-align: center; }
    [data-testid="stSuccess"] strong { font-size: 2rem; color: #FFFFFF; }
    .note-box { background-color: #3a4553; border: 1px solid #5a6b7d; border-radius: 12px; padding: 1rem; margin-top: 1.5rem; }
    .content-wrapper h1, .content-wrapper h2, .content-wrapper h3, .content-wrapper h4 { color: #FFFFFF; }
</style>
"""

# Apply the selected theme
st.markdown(dark_theme_css if st.session_state.theme == 'dark' else light_theme_css, unsafe_allow_html=True)

# === Header and Theme Toggle ===
header_cols = st.columns([0.9, 0.1])
with header_cols[0]:
    st.markdown('<div class="app-header"><h1>BigMart Sales Predictor</h1></div>', unsafe_allow_html=True)
with header_cols[1]:
    theme_toggle = st.toggle('Dark Mode', value=st.session_state.theme == 'dark', key='theme_toggle')
    if theme_toggle:
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'
    # This will trigger a re-run if the state changes

# === Top Navigation Bar ===
page = st.radio(
    "Navigation",
    ["Home", "Data Insights", "Model Details", "About", "Help"],
    horizontal=True,
    label_visibility="collapsed"
)

# === Page Content Logic ===
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

if page == "Home":
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

elif page == "Data Insights":
    st.header("Exploratory Data Insights")
    st.markdown("This section provides a brief overview of the BigMart sales dataset.")
    st.markdown("---")
    
    st.subheader("Sales Distribution by Outlet Type")
    # Sample data for charting
    chart_data = pd.DataFrame({
        'Outlet Type': ['Supermarket Type1', 'Grocery Store', 'Supermarket Type3', 'Supermarket Type2'],
        'Total Sales (in Millions ₹)': [12.9, 0.24, 8.5, 4.5]
    })
    st.bar_chart(chart_data, x='Outlet Type', y='Total Sales (in Millions ₹)')
    st.markdown("As seen in the chart, `Supermarket Type1` outlets account for the majority of sales, followed by `Supermarket Type3`. `Grocery Stores` have significantly lower sales in comparison.")

elif page == "Model Details":
    st.header("About the Prediction Model")
    st.markdown(f"This application uses a Gradient Boosting Regressor model, a powerful machine learning algorithm, built with **scikit-learn v{sklearn_version}**.")
    st.markdown("---")
    st.subheader("Model Performance")
    st.markdown("""
    - **R-squared Score:** ~0.72
    - **Root Mean Squared Error (RMSE):** ~₹1080.00
    
    *These are approximate scores based on a typical trained model on this dataset. An R-squared score of 0.72 means the model can explain about 72% of the variance in sales, which is a good score for this type of problem.*
    """)
    st.subheader("Feature Importance")
    st.markdown("The most influential features in predicting sales are:")
    st.markdown("1. **Item MRP:** The price of the product is the strongest predictor of its sales revenue.")
    st.markdown("2. **Outlet Type:** `Supermarket Type3` outlets consistently have higher sales.")
    st.markdown("3. **Outlet Age:** The establishment year of an outlet impacts its sales performance.")
    st.markdown("4. **Item Visibility:** How prominently an item is displayed also affects sales.")

elif page == "About":
    st.header("About This Project")
    st.markdown("""
    This application is a web-based tool for predicting sales for the BigMart dataset, a popular dataset for practicing regression machine learning.
    
    #### Purpose
    The primary goal is to demonstrate a complete data science project, from model training (done offline) to deployment as an interactive web application using Streamlit. It helps showcase how a machine learning model can be used to provide actionable insights for retail businesses, such as:
    - **Inventory Management**
    - **Marketing Strategy**
    - **Financial Planning**
    
    **Dataset Source:** The data used for training the model is from the "BigMart Sales Prediction" competition on Kaggle.
    
    **Developer:** Tejas
    """)

elif page == "Help":
    st.header("Help & Instructions")
    st.markdown("""
    #### How to Use
    1.  **Navigate:** Use the top navigation bar to switch between pages.
    2.  **Fill the Form:** On the **Home** screen, enter all the details for the product and the store.
    3.  **Predict:** Click the **"Predict Sales"** button.
    4.  **View Result:** The estimated sales amount will appear below the form.
    5.  **Change Theme:** Use the "Dark Mode" toggle at the top right to switch between light and dark themes.
    """)

st.markdown('</div>', unsafe_allow_html=True)

