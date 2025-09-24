import streamlit as st
import pandas as pd
import pickle

# === App Configuration ===
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === Load Model and Version Info from Pickle ===
try:
    with open("bigmart_best_model.pkl", "rb") as f:
        model, sklearn_version = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'bigmart_best_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()

# === Custom CSS for Mobile App Theme ===
st.markdown("""
<style>
    /* --- General App Body & Layout --- */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        background-color: #EAEAF2; /* Light grey background to frame the app */
        color: #111;
    }

    /* --- Main App Container --- */
    /* This targets the main block container from Streamlit */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    /* We create our own mobile-like container */
    .app-container {
        max-width: 480px; /* Typical mobile screen width */
        margin: 0 auto;
        background-color: #F8F8FF; /* Off-white for the app background */
        min-height: 100vh;
        border-left: 1px solid #dcdce0;
        border-right: 1px solid #dcdce0;
        box-shadow: 0 0 20px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
    }
    
    .app-content {
        padding: 1.5rem 1.5rem 8rem 1.5rem; /* Bottom padding to avoid overlap with nav bar */
        flex-grow: 1;
    }

    /* --- Sticky Header --- */
    .app-header {
        position: sticky;
        top: 0;
        background-color: #F8F8FF;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e0e0e0;
        z-index: 10;
        text-align: center;
    }
    .app-header h2 {
        margin: 0;
        font-size: 1.5rem;
        color: #2874f0;
    }

    /* --- Input Widgets Styling --- */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        font-weight: 600;
        color: #333;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1px solid #c2c2c2;
        box-shadow: none;
        background-color: #fff;
        padding: 1rem 0.75rem;
        font-size: 1.1rem;
        color: #333;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #2874f0;
        box-shadow: 0 0 0 2px rgba(40, 116, 240, 0.2);
    }

    /* --- Slider Styling --- */
    .stSlider {
        padding-top: 0.5rem;
    }
    .stSlider [data-baseweb="slider"] {
        padding-bottom: 1rem;
    }
    .stSlider .stThumb {
        background-color: #2874f0;
        height: 24px;
        width: 24px;
        border: 4px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stSlider .stTrack {
        background-color: #c2c2c2;
        height: 8px;
    }
    
    /* --- Button Styling --- */
    .stFormSubmitButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 1rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        background: linear-gradient(45deg, #fb641b, #f99f2e);
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 12px rgba(251, 100, 27, 0.3);
    }
    .stFormSubmitButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(251, 100, 27, 0.4);
    }
    .stFormSubmitButton > button:active {
        transform: scale(0.99);
    }
    
    /* --- Result & Note Box Styling --- */
    [data-testid="stSuccess"] {
        background-color: #e6ffed;
        border: 1px solid #23c552;
        border-radius: 12px;
        padding: 1.5rem;
        color: #111;
        text-align: center;
    }
    [data-testid="stSuccess"] strong {
        font-size: 1.8rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    .note-box {
        background-color: #e6f7ff;
        border: 1px solid #91d5ff;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #333;
    }
    
    /* --- Bottom Navigation Bar --- */
    div[role="radiogroup"] {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        max-width: 480px; /* Match the app container width */
        margin: 0 auto; /* Center the nav bar */
        display: flex;
        justify-content: space-around;
        padding: 0.75rem 0.5rem;
        background-color: #ffffff;
        border-top: 1px solid #e0e0e0;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        z-index: 100;
    }
    div[role="radiogroup"] label {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.25rem;
        cursor: pointer;
        transition: color 0.2s ease-in-out;
        color: #878787;
        font-weight: 500;
        font-size: 0.75rem;
        flex-grow: 1;
    }
    div[role="radiogroup"] input[type="radio"] {
        display: none; /* Hide the actual radio button */
    }
    div[role="radiogroup"] label svg {
        width: 24px;
        height: 24px;
    }
    div[role="radiogroup"] label:has(input:checked) {
        color: #2874f0; /* Active link color */
    }
    div[role="radiogroup"] label:has(input:checked) svg {
        fill: #2874f0;
    }

    /* --- Other Elements --- */
    hr {
        margin: 1.5rem 0;
    }
    .stSpinner {
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Icons for Bottom Navigation
home_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#878787"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>"""
about_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#878787"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>"""
help_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#878787"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/></svg>"""

# === Main App Structure ===
st.markdown('<div class="app-container">', unsafe_allow_html=True)

# This will be filled by the page content based on navigation
page_container = st.container()

# === Bottom Navigation Bar ===
page = st.radio(
    "Navigation",
    [
        f"{home_icon}Home",
        f"{about_icon}About",
        f"{help_icon}Help"
    ],
    horizontal=True,
    label_visibility="collapsed",
    format_func=lambda x: x.split("</svg>")[-1] # Show only the text label
)

# === Page Content Logic ===
with page_container:
    # --- HOME PAGE ---
    if "Home" in page:
        st.markdown('<div class="app-header"><h2>Sales Predictor</h2></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-content">', unsafe_allow_html=True)

        with st.form("prediction_form"):
            st.selectbox("🍔 Item Fat Content", ["Low Fat", "Regular"], key="Item_Fat_Content")
            st.selectbox("📦 Item Type", [
                "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
                "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
                "Health and Hygiene", "Hard Drinks", "Canned", "Breads",
                "Starchy Foods", "Others", "Seafood"
            ], key="Item_Type")
            st.number_input("💵 Item MRP (₹)", min_value=0.0, value=150.0, key="Item_MRP")
            st.slider("👁️ Item Visibility", min_value=0.0, max_value=0.35, step=0.01, value=0.05, key="Item_Visibility")
            st.number_input("⚖️ Item Weight (kg)", min_value=0.0, value=12.5, key="Item_Weight")
            
            st.markdown("---")

            st.selectbox("🏪 Outlet Identifier", [
                "OUT027", "OUT013", "OUT049", "OUT035", "OUT046",
                "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
            ], key="Outlet_Identifier")
            st.selectbox("📏 Outlet Size", ["Small", "Medium", "High"], key="Outlet_Size")
            st.selectbox("🌍 Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"], key="Outlet_Location_Type")
            st.selectbox("🏬 Outlet Type", [
                "Supermarket Type1", "Supermarket Type2",
                "Supermarket Type3", "Grocery Store"
            ], key="Outlet_Type")
            st.slider("📅 Outlet Age (Years)", 0, 40, 15, key="Outlet_Age")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Predict Sales")

        if submit:
            input_df = pd.DataFrame([{
                "Item_Identifier": "FAD123", # Placeholder, as it's not used by most models after encoding
                "Item_Weight": st.session_state.Item_Weight,
                "Item_Fat_Content": st.session_state.Item_Fat_Content,
                "Item_Visibility": st.session_state.Item_Visibility,
                "Item_Type": st.session_state.Item_Type,
                "Item_MRP": st.session_state.Item_MRP,
                "Outlet_Identifier": st.session_state.Outlet_Identifier,
                "Outlet_Size": st.session_state.Outlet_Size,
                "Outlet_Location_Type": st.session_state.Outlet_Location_Type,
                "Outlet_Type": st.session_state.Outlet_Type,
                "Outlet_Age": st.session_state.Outlet_Age
            }])
            
            with st.spinner('Predicting...'):
                try:
                    prediction = model.predict(input_df)[0]
                    st.success(f"Predicted Sales: **₹{prediction:,.2f}**")
                    st.markdown("""
                    <div class="note-box">
                        <b>Note:</b> This is an estimate based on historical data. Actual sales can vary due to promotions, seasonality, etc.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ABOUT PAGE ---
    elif "About" in page:
        st.markdown('<div class="app-header"><h2>About</h2></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-content">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 Purpose
        This app provides a sales prediction for the BigMart dataset using a machine learning model. It helps managers forecast sales to optimize inventory and marketing strategies.

        ### 🛠️ How It Works
        The prediction is generated by a pre-trained regression model that considers various product and store features.
        
        - **Product Features:** Item weight, fat content, visibility, type, and MRP.
        - **Store Features:** Outlet size, location type, store type, and age.

        This tool showcases how data science can be applied to solve real-world retail challenges.
        
        **Model Version:** scikit-learn v{sklearn_version}
        
        **Developer:** Tejas
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- HELP PAGE ---
    elif "Help" in page:
        st.markdown('<div class="app-header"><h2>Help</h2></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-content">', unsafe_allow_html=True)
        st.markdown("""
        ### 📝 How to Use
        1.  **Fill the Form:** On the **Home** screen, enter all the details for the product and the store.
        2.  **Adjust Sliders:** Use the sliders for `Item Visibility` and `Outlet Age`.
        3.  **Predict:** Tap the **"Predict Sales"** button.
        4.  **View Result:** The estimated sales amount will appear at the bottom.

        ### 🤔 FAQ
        **Q: How accurate is the prediction?**
        
        A: The prediction is a data-driven estimate. Real-world sales can be influenced by many factors not included in the model, like holidays or special promotions.

        **Q: Why is "Item Identifier" not an input?**
        
        A: The model was trained in a way that it doesn't require the specific item ID for prediction, as it learns from the other features provided.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Close the main app container div
st.markdown('</div>', unsafe_allow_html=True)
