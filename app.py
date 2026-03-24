import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== 1. การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="🛍️",
    layout="centered"
)

# ===== 2. ฟังก์ชันโหลดโมเดล =====
@st.cache_resource
def load_retail_model():
    try:
        # โหลดไฟล์ pkl ที่คุณมี
        model = joblib.load("retail_sales_model.pkl")
        return model
    except Exception as e:
        return e

model = load_retail_model()

# ===== 3. ส่วนการแสดงผลหน้าเว็บ =====
st.title("📊 ระบบทำนายยอดขายรายสินค้า")
st.write("กรอกข้อมูลด้านล่างเพื่อประมวลผลการทำนายยอดขาย")
st.divider()

if isinstance(model, Exception):
    st.error(f"ไม่สามารถโหลดโมเดลได้: {model}")
    st.info("กรุณาตรวจสอบว่ามีไฟล์ 'retail_sales_model.pkl' อยู่ในโฟลเดอร์เดียวกันหรือไม่")
else:
    # สร้างฟอร์มรับข้อมูล
    with st.form("sales_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📦 ข้อมูลสต็อกและคำสั่งซื้อ")
            inventory = st.number_input("Inventory Level", value=500)
            units_ordered = st.number_input("Units Ordered", value=50)
            demand_forecast = st.number_input("Demand Forecast", value=100)
            category = st.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Health & Beauty", "Toys"])
            store_id = st.text_input("Store ID", "S001")
            product_id = st.text_input("Product ID", "P001")

        with col2:
            st.subheader("💰 ข้อมูลราคาและโปรโมชั่น")
            price = st.number_input("Price", value=250.0)
            discount = st.slider("Discount (%)", 0, 100, 10)
            comp_pricing = st.number_input("Competitor Pricing", value=245.0)
            holiday_promo = st.selectbox("Holiday Promotion", [0, 1], format_func=lambda x: "มีโปร" if x == 1 else "ไม่มีโปร")

        st.subheader("🌍 ข้อมูลพื้นที่และสภาพอากาศ")
        c3, c4, c5 = st.columns(3)
        with c3:
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        with c4:
            weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy"])
        with c5:
            season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

        submit = st.form_submit_button("🔮 ทำนายยอดขายทันที")

    # ===== 4. ส่วนการทำนายผล =====
    if submit:
        # เตรียมข้อมูลให้อยู่ในรูปแบบ DataFrame
        # ชื่อ Column ต้องสะกดตรงกับที่ใช้เทรนโมเดลเป๊ะๆ
        input_dict = {
            "Inventory_Level": inventory,
            "Units_Ordered": units_ordered,
            "Demand_Forecast": demand_forecast,
            "Price": price,
            "Discount": discount,
            "Competitor_Pricing": comp_pricing,
            "Holiday_Promotion": holiday_promo,
            "Store_ID": store_id,
            "Product_ID": product_id,
            "Category": category,
            "Region": region,
            "Weather_Condition": weather,
            "Seasonality": season
        }
        
        df_input = pd.DataFrame([input_dict])

        try:
            # คำนวณยอดขาย
            result = model.predict(df_input)[0]
            
            # แสดงผล
            st.markdown("---")
            st.balloons()
            st.success("### วิเคราะห์ผลสำเร็จ")
            st.metric(label="ยอดขายที่คาดการณ์ (Units/Value)", value=f"{max(0, result):,.2f}")
            
            with st.expander("ดูข้อมูลชุดที่ใช้ทำนาย"):
                st.write(df_input)
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
            st.info("สาเหตุอาจเกิดจากเวอร์ชันของ scikit-learn ในเครื่องไม่ตรงกับที่ใช้สร้างโมเดล")

st.caption("Retail Sales Prediction Tool v1.0")
