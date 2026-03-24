# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="ระบบพยากรณ์ยอดขายสินค้า",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        return joblib.load("retail_sales_model.pkl")
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่ได้: {e}")
        return None

model = load_model()

st.title("📊 Retail Sales Prediction")
st.markdown("ระบบวิเคราะห์และพยากรณ์ยอดขายด้วย AI")
st.divider()

if model is None:
    st.stop()

with st.form("form"):
    st.subheader("📝 กรอกข้อมูล")

    col1, col2 = st.columns(2)

    with col1:
        inventory = st.number_input("Inventory Level", 0, 10000, 500)
        units = st.number_input("Units Ordered", 0, 1000, 50)
        demand = st.number_input("Demand Forecast", 0, 2000, 100)
        price = st.number_input("Price", 0.0, 10000.0, 250.0)
        discount = st.slider("Discount (%)", 0, 100, 10)
        comp = st.number_input("Competitor Pricing", 0.0, 10000.0, 245.0)
        promo = st.selectbox("Holiday Promotion", [0, 1])

    with col2:
        store = st.text_input("Store ID", "S001")
        product = st.text_input("Product ID", "P001")
        category = st.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Health & Beauty", "Toys"])
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy"])
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

    submit = st.form_submit_button("🔮 ทำนาย")

if submit:
    input_df = pd.DataFrame([{
        "Inventory_Level": inventory,
        "Units_Ordered": units,
        "Demand_Forecast": demand,
        "Price": price,
        "Discount": discount,
        "Competitor_Pricing": comp,
        "Holiday_Promotion": promo,
        "Store_ID": store,
        "Product_ID": product,
        "Category": category,
        "Region": region,
        "Weather_Condition": weather,
        "Seasonality": season
    }])

    try:
        with st.spinner("กำลังวิเคราะห์..."):
            pred = model.predict(input_df)[0]

        pred = float(max(0, pred))

        st.success("✅ วิเคราะห์เสร็จ")
        st.metric("ยอดขายที่คาดการณ์", f"{pred:,.2f}")

        if pred > 500:
            st.info("📈 แนวโน้มขายดี")
        else:
            st.warning("📉 ควรเพิ่มโปรโมชั่น")

        with st.expander("ดูข้อมูล"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"❌ Error: {e}")
