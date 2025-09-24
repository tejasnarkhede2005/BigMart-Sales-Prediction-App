import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# === App Configuration ===
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Load Model and Version Info from Pickle ===
try:
    with open("bigmart_best_model.pkl", "rb") as f:
        model, sklearn_version = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'bigmart_best_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()

# === THEME CSS ===
dark_theme_css = """
<style>
    /* Custom Title Style */
    .app-title {
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 1px;
    }

    body { background-color: #131921; color: #FFFFFF; }
    .block-container { padding: 2rem 3rem 3rem 3rem !important; }
    
    /* Removed the .content-wrapper class as it's no longer used */

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
</style>
"""

# Apply the dark theme CSS
st.markdown(dark_theme_css, unsafe_allow_html=True)

# === Header ===
st.markdown('<h1 class="app-title">🛒 BigMart Sales Predictor</h1>', unsafe_allow_html=True)

# === Top Navigation Bar (Centered) ===
_, nav_col, _ = st.columns([0.2, 0.6, 0.2])
with nav_col:
    page = st.radio(
        "Navigation",
        ["Home", "Data Insights", "Model Details", "About", "Contact"],
        horizontal=True,
        label_visibility="collapsed"
    )

# === Page Content Logic ===
# The <div class="content-wrapper"> has been removed.

if page == "Home":
    form_cols = st.columns([0.2, 0.6, 0.2])
    with form_cols[1]:
        st.header("Sales Prediction Form")
        st.markdown("Fill in the details below to generate a sales prediction.")
        st.markdown("---")

        with st.form("prediction_form"):
            st.selectbox("🍔 Item Fat Content", ["Low Fat", "Regular"], key="Item_Fat_Content")
            st.selectbox("📦 Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"], key="Item_Type")
            st.number_input("💵 Item MRP (₹)", min_value=0.0, value=150.0, key="Item_MRP")
            st.slider("👁️ Item Visibility", min_value=0.0, max_value=0.35, step=0.01, value=0.05, key="Item_Visibility")
            st.number_input("⚖️ Item Weight (kg)", min_value=0.0, value=12.5, key="Item_Weight")
            st.markdown("---")
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
                    st.markdown("""<div class="note-box"><b>Note:</b> This is an estimate based on historical data.</div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

elif page == "Data Insights":
    st.header("Exploratory Data Insights")
    st.markdown("This section provides a brief overview of the BigMart sales dataset.")
    st.markdown("---")
    
    chart_data = pd.DataFrame({
        'Outlet Type': ['Supermarket Type1', 'Grocery Store', 'Supermarket Type3', 'Supermarket Type2'],
        'Total Sales (in Millions ₹)': [12.9, 0.24, 8.5, 4.5]
    })

    # Set chart colors for the dark theme
    font_color = "#FFFFFF"
    bar_color = "#FF9900"

    # Create a Plotly figure
    fig = px.bar(
        chart_data,
        x='Outlet Type',
        y='Total Sales (in Millions ₹)'
    )
    
    # Update layout for theme and horizontal labels
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=font_color,
        xaxis_tickangle=0, 
        title={
            'text': "Sales Distribution by Outlet Type",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    # Update trace for bar color
    fig.update_traces(marker_color=bar_color)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("`Supermarket Type1` outlets account for the majority of sales, followed by `Supermarket Type3`. `Grocery Stores` have significantly lower sales.")

elif page == "Model Details":
    st.header("About the Prediction Model")
    st.markdown(f"This application uses a Gradient Boosting Regressor model built with **scikit-learn v{sklearn_version}**.")
    st.markdown("---")
    st.subheader("Model Performance")
    st.markdown("- **R-squared Score:** ~0.72\n- **Root Mean Squared Error (RMSE):** ~₹1080.00")
    st.markdown("*An R-squared score of 0.72 means the model can explain about 72% of the variance in sales.*")
    st.subheader("Feature Importance")
    st.markdown("1. **Item MRP:** The price is the strongest predictor of sales revenue.\n2. **Outlet Type:** `Supermarket Type3` outlets consistently have higher sales.\n3. **Outlet Age:** The establishment year impacts sales performance.")

elif page == "About":
    st.header("About This Project")
    st.markdown("""
    This application is a web-based tool for predicting sales for the BigMart dataset, a popular dataset for practicing regression machine learning.
    #### Purpose
    The primary goal is to demonstrate a complete data science project, from model training (done offline) to deployment as an interactive web application. 
    It helps showcase how a machine learning model can be used to provide actionable insights for retail businesses.
    
    **Developer:** Tejas
    """)

elif page == "Contact":
    st.header("Contact Information")
    st.markdown("For any inquiries, feedback, or issues with the application, please reach out.")
    st.markdown("---")
    st.markdown("- **Developer:** Tejas")
    st.markdown("- **Email:** tejasnarkhede03@gmail.com")
    st.markdown("- **GitHub:** [github.com/tejasnarkhede2005](https://github.com/tejasnarkhede2005)")

