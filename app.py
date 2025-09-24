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

# === Load Model and Version Info from Pickle ===
try:
    with open("bigmart_best_model.pkl", "rb") as f:
        model, sklearn_version = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'bigmart_best_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()

# === Custom CSS for Flipkart Theme ===
st.markdown("""
<style>
    /* General Styles - Flipkart Theme */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        background-color: #f1f3f6; /* Flipkart's light grey background */
        color: #333;
    }

    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Streamlit Title and Headers */
    h1, h2, h3 {
        color: #2874f0; /* Flipkart Blue */
        font-weight: 600;
    }
    
    /* Card/Container for the form and pages */
    .content-container {
        background-color: #ffffff; /* White background for content */
        padding: 2.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        border: 1px solid #e0e0e0;
    }

    /* Input Widgets Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 5px;
        border: 1px solid #c2c2c2;
        box-shadow: none;
        background-color: #fff;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #2874f0; /* Highlight with Flipkart blue on focus */
        box-shadow: 0 0 0 2px rgba(40, 116, 240, 0.2);
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        padding: 0.5rem 0;
    }
    .stSlider .stThumb {
        background-color: #2874f0; /* Flipkart blue for slider thumb */
    }
    .stSlider .stTrack {
        background-color: #c2c2c2;
    }

    /* Button Styling */
    .stButton > button, .stFormSubmitButton > button {
        width: 100%;
        border-radius: 5px;
        border: none;
        padding: 0.8rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        background-color: #fb641b; /* Flipkart Orange for buttons */
        transition: background-color 0.3s, box-shadow 0.3s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #e1540f; /* Darker orange on hover */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Success Box Styling */
    [data-testid="stSuccess"] {
        background-color: #e6ffed;
        border-left: 5px solid #23c552;
        border-radius: 8px;
        padding: 1rem;
        color: #222;
    }
    [data-testid="stError"] {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4d4d;
        border-radius: 8px;
        padding: 1rem;
        color: #222;
    }

    /* Note Box Styling */
    .note-box {
        background-color: #e6f7ff;
        border: 1px solid #91d5ff;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #333;
    }
    
    /* Footer */
    footer {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
    }

    /* Navbar styling using Streamlit's st.radio */
    div[role="radiogroup"] {
        flex-direction: row;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[role="radiogroup"] label {
        padding: 0.5rem 1.5rem;
        border: 1px solid transparent;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        font-weight: 500;
        font-size: 1.1rem;
    }
    /* Hide the actual radio circle */
    div[role="radiogroup"] input[type="radio"] {
        display: none;
    }
    /* Style for the selected/active link */
    div[role="radiogroup"] label:has(input:checked) {
        background-color: #2874f0;
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(40, 116, 240, 0.3);
    }
    /* Style for hover effect */
    div[role="radiogroup"] label:not(:has(input:checked)):hover {
        background-color: #e6f0ff;
        color: #2874f0;
    }
</style>
""", unsafe_allow_html=True)

# === Header and Navigation Bar ===
st.markdown('<h1 style="text-align: center; color: #2874f0;">🛒 BigMart Sales Predictor</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align: center;">Powered by a scikit-learn v{sklearn_version} model</p>', unsafe_allow_html=True)

page = st.radio(
    "Navigation",
    ["Home", "About", "Help"],
    label_visibility="collapsed" # Hides the "Navigation" label
)

# === Page Content ===
if page == "Home":
    # === Create Form for Inputs ===
    with st.container():
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            st.header("🧾 Enter Item & Outlet Details")
            st.markdown("Fill in the fields below to get a sales prediction.")
            st.markdown("---")
            
            col1, col2 = st.columns(2)

            with col1:
                Item_Identifier = st.text_input("🆔 Item Identifier", "FDA15", help="Unique ID for the product.")
                Item_Weight = st.number_input("⚖️ Item Weight (kg)", min_value=0.0, value=12.5, help="Weight of the product.")
                Item_Fat_Content = st.selectbox("🍔 Item Fat Content", ["Low Fat", "Regular"], help="Fat content category.")
                Item_Visibility = st.slider("👁️ Item Visibility", min_value=0.0, max_value=0.35, step=0.01, value=0.05, help="Display area percentage in the store.")
                Item_Type = st.selectbox("📦 Item Type", [
                    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
                    "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
                    "Health and Hygiene", "Hard Drinks", "Canned", "Breads",
                    "Starchy Foods", "Others", "Seafood"
                ], help="Category of the product.")
                Item_MRP = st.number_input("💵 Item MRP (₹)", min_value=0.0, value=150.0, help="Maximum Retail Price of the product.")

            with col2:
                Outlet_Identifier = st.selectbox("🏪 Outlet Identifier", [
                    "OUT027", "OUT013", "OUT049", "OUT035", "OUT046",
                    "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
                ], help="Unique ID for the store.")
                Outlet_Size = st.selectbox("📏 Outlet Size", ["Small", "Medium", "High"], help="Size of the store.")
                Outlet_Location_Type = st.selectbox("🌍 Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"], help="Type of city where the store is located.")
                Outlet_Type = st.selectbox("🏬 Outlet Type", [
                    "Supermarket Type1", "Supermarket Type2",
                    "Supermarket Type3", "Grocery Store"
                ], help="Format of the store.")
                Outlet_Age = st.slider("📅 Outlet Age (Years)", 0, 40, 15, help="Number of years the store has been operational.")

            # === Submit Button ===
            st.markdown("<br>", unsafe_allow_html=True)
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
            with st.spinner('Calculating...'):
                prediction = model.predict(input_df)[0]
                st.success(f"📈 **Predicted Item Outlet Sales:** ₹{prediction:,.2f}")

            st.markdown("""
            <div class="note-box">
                <b>🧠 Note:</b> This prediction is based on historical sales data. Actual sales may vary due to external factors like promotions, seasonality, and local market conditions.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ **Prediction failed:** An error occurred. Please check your inputs. Details: {e}")

elif page == "About":
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.header("📖 About This Project")
    st.markdown("""
    This application is a sales prediction tool for the **BigMart** dataset. It demonstrates the use of a machine learning model to forecast product sales in various retail stores.

    ### 🎯 Purpose
    The primary goal is to provide store managers and planners with a reliable estimate of sales for a given product in a specific store. This can help in:
    - **Inventory Management:** Optimizing stock levels to prevent overstocking or stockouts.
    - **Marketing Strategy:** Understanding which products perform best in different store types and locations.
    - **Financial Planning:** Forecasting revenue and making informed business decisions.

    ### 🛠️ How It Works
    The prediction is generated by a regression model trained on thousands of historical sales records. The model considers various features of both the product and the store, such as:
    - **Product Features:** Item weight, fat content, visibility, type, and MRP.
    - **Store Features:** Outlet size, location type, store type, and age.
    
    The model has been pre-processed and tuned to achieve accurate predictions based on these inputs.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Help":
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.header("❓ Help & Instructions")
    st.markdown("""
    Welcome to the Help page! Here’s how to use the BigMart Sales Predictor.

    ### 📝 How to Get a Prediction
    1.  **Navigate to the 'Home' page** using the navigation bar at the top.
    2.  **Fill in all the fields** in the form with the details of the product and the store.
    3.  **Use the sliders** for `Item Visibility` and `Outlet Age` to select a value.
    4.  **Click the "Predict Sales" button** at the bottom of the form.
    5.  The predicted sales amount (in ₹) will be displayed in a green box below the form.

    ### 🤔 Frequently Asked Questions (FAQ)

    **Q: What do the input fields mean?**
    - **Item Identifier:** A unique code for each product (e.g., `FDA15`).
    - **Item Weight:** The physical weight of the product in kilograms.
    - **Item Fat Content:** Whether the product is 'Low Fat' or 'Regular'.
    - **Item Visibility:** The percentage of the total display area in a store allocated to this specific product.
    - **Item MRP:** The Maximum Retail Price of the product.
    - **Outlet Identifier:** A unique code for each store (e.g., `OUT027`).
    - **Outlet Size:** The size of the store, categorized as 'Small', 'Medium', or 'High'.
    - **Outlet Location Type:** The tier of the city where the store is located (Tier 1, 2, or 3).
    - **Outlet Type:** The format of the store, like 'Supermarket' or 'Grocery Store'.
    - **Outlet Age:** The number of years the store has been in operation.

    **Q: How accurate is the prediction?**
    - The prediction is based on a machine learning model and provides a calculated estimate. While it is built for accuracy, real-world sales can be influenced by many factors not included in the model, such as holidays, special promotions, or local events.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown("<footer>Developed with ❤️ using Streamlit | Built by Tejas</footer>", unsafe_allow_html=True)
