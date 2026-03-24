import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="Retail Sales Prediction System",
    page_icon="🛍️",
    layout="centered"
)

# ===== โหลดโมเดล =====
import streamlit as st
import joblib
import sklearn

# ตรวจสอบเวอร์ชันเพื่อ debug (จะแสดงบนหน้าเว็บ)
# st.write(f"Current sklearn version: {sklearn.__version__}")

@st.cache_resource
def load_model():
    try:
        # โหลดโมเดลตรงๆ
        return joblib.load("retail_sales_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("คำแนะนำ: ตรวจสอบว่า scikit-learn ในไฟล์ requirements.txt เป็นเวอร์ชัน 1.6.1")
        return None

model = load_model()

if model:
    st.success("✅ โหลดโมเดลสำเร็จ! พร้อมใช้งาน")
    # ... ส่วนของ Form รับค่า และ model.predict ตามที่ให้ไว้ก่อนหน้านี้ ...

# ===== ส่วนหัวของเว็บไซต์ =====
st.title("📊 ระบบพยากรณ์ยอดขายสินค้า")
st.markdown("กรุณากรอกข้อมูลรายละเอียดสินค้าและสภาวะตลาดเพื่อทำนายยอดขาย")
st.divider()

# ===== สร้าง Form รับข้อมูล =====
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 ข้อมูลสินค้า")
        inventory = st.number_input("Inventory Level", min_value=0, value=500)
        units_ordered = st.number_input("Units Ordered", min_value=0, value=50)
        demand_forecast = st.number_input("Demand Forecast", min_value=0, value=100)
        category = st.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Health & Beauty", "Toys"])

    with col2:
        st.subheader("💰 ข้อมูลราคาและโปรโมชั่น")
        price = st.number_input("Price", min_value=0.0, value=250.0)
        discount = st.slider("Discount (%)", 0, 100, 10)
        comp_pricing = st.number_input("Competitor Pricing", min_value=0.0, value=245.0)
        holiday_promo = st.selectbox("Holiday Promotion", [0, 1], format_func=lambda x: "มีโปรโมชั่น" if x == 1 else "ไม่มี")

    st.subheader("🌍 ปัจจัยภายนอก")
    c3, c4, c5 = st.columns(3)
    with c3:
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
    with c4:
        weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Stormy"])
    with c5:
        season = st.selectbox("Seasonality", ["Spring", "Summer", "Autumn", "Winter"])

    # ปุ่ม Submit
    submit_button = st.form_submit_button(label="🔮 ทำนายยอดขาย")

# ===== ส่วนการประมวลผล =====
if submit_button:
    # 1. จัดเตรียมข้อมูลให้ตรงกับ Column ที่โมเดลต้องการ
    input_data = pd.DataFrame([{
        "Inventory_Level": inventory,
        "Units_Ordered": units_ordered,
        "Demand_Forecast": demand_forecast,
        "Price": price,
        "Discount": discount,
        "Competitor_Pricing": comp_pricing,
        "Holiday_Promotion": holiday_promo,
        "Store_ID": "S001",  # ใส่ค่า Default หรือเพิ่ม Input ได้
        "Product_ID": "P001", # ใส่ค่า Default หรือเพิ่ม Input ได้
        "Category": category,
        "Region": region,
        "Weather_Condition": weather,
        "Seasonality": season
    }])

    try:
        # 2. ทำนายผล
        prediction = model.predict(input_data)[0]

        # 3. แสดงผลลัพธ์
        st.success("### ผลการวิเคราะห์")
        st.metric(label="ยอดขายที่คาดการณ์ (Sales Prediction)", value=f"{prediction:,.2f} บาท")
        
        # แสดงกราฟเปรียบเทียบง่ายๆ
        st.progress(min(prediction / 10000, 1.0), text="ระดับความหนาแน่นของยอดขาย")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")

# ===== คำแนะนำเพิ่มเติม =====
with st.expander("ℹ️ ข้อมูลเพิ่มเติมเกี่ยวกับโมเดล"):
    st.write("""
    - โมเดลนี้ถูกเทรนด้วย `Scikit-learn Pipeline`
    - รองรับการทำ Preprocessing ข้อมูลหมวดหมู่ (Categorical) และตัวเลข (Numerical) อัตโนมัติ
    - ข้อมูลเบื้องต้นอ้างอิงจากฟีเจอร์ที่ตรวจพบในไฟล์ .pkl ของคุณ
    """)
