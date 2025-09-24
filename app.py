import streamlit as st
import pandas as pd
import pickle

# === Load Model and Version Info from Pickle ===
with open("bigmart_best_model.pkl", "rb") as f:
    model, sklearn_version = pickle.load(f)

# === App Configuration ===
st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒", layout="centered")

# === Title & Introduction ===
st.title("🛒 BigMart Sales Prediction App")
st.markdown(f"""
Welcome to the **BigMart Sales Prediction App**!  
This tool uses a machine learning model trained using **scikit-learn v{sklearn_version}** to predict item sales based on product and store characteristics.

📌 *Fill in the item and outlet details below, and click Predict to see the estimated sales.*
""")

# === Form Header ===
st.markdown("---")
st.header("🧾 Enter Item & Outlet Details")

# === Create Form for Inputs ===
with st.form("prediction_form"):
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

    st.markdown("### 🔍 Input Summary")
    st.dataframe(input_df)

    try:
        prediction = model.predict(input_df)[0]
        st.markdown("---")
        st.success(f"📈 **Predicted Item Outlet Sales:** ₹{prediction:,.2f}")

        st.markdown(f"""
        <div style="background-color:#f0f8ff;padding:10px;border-radius:5px;">
        <b>🧠 Note:</b> This prediction is based on historical sales data and outlet characteristics. Actual sales may vary due to external factors like promotions, seasonality, etc.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# === Expandable Info Section ===
with st.expander("📘 About this Model"):
    st.markdown("""
    This sales prediction model was trained on the **BigMart** dataset using various regression algorithms. 
    The final selected model offers robust performance and was fine-tuned using:
    - Feature engineering (like converting year to outlet age)
    - One-hot encoding of categorical variables
    - Normalization of visibility
    - Hyperparameter tuning

    **Target Variable:** Item_Outlet_Sales (₹)  
    **Model Type:** Regression  
    **Use Case:** Forecasting sales to optimize stock and marketing strategies.
    """)

# === Footer ===
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Model by Maruti")
