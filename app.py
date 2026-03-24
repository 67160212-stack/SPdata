import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

# ===== 1. การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="Retail Sales Prediction System",
    page_icon="🛍️",
    layout="centered"
)

# ===== 2. ฟังก์ชันโหลดโมเดล =====
@st.cache_resource
def load_model():
    try:
        # โหลดโมเดลตรงๆ จากไฟล์ที่อัปโหลด
        return joblib.load("retail_sales_model.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 คำแนะนำ: ตรวจสอบว่าในไฟล์ requirements.txt ระบุ scikit-learn==1.6.1 และ Python ใน Settings เป็น 3.11 หรือ 3.12")
        return None

model = load_model()

# ===== 3. ส่วนหัวของเว็บไซต์ =====
st.title("📊 ระบบพยากรณ์ยอดขายสินค้า")
st.markdown("กรุณากรอกข้อมูลรายละเอียดสินค้าและสภาวะตลาดเพื่อทำนายยอดขาย")

if model:
    st.success("✅ เชื่อมต่อโมเดลสำเร็จ!")
    st.divider()

    # ===== 4. สร้าง Form รับข้อมูล =====
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📦 ข้อมูลสินค้า")
            inventory = st.number_input("Inventory Level", min_value=0, value=500)
            units_ordered = st.number_input("Units Ordered", min_value=0, value=50)
            demand_forecast = st.number_input("Demand Forecast", min_value=0, value=100)
            category = st.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Health & Beauty", "Toys"])
            store_id = st.text_input("Store ID", "S001")
            product_id = st.text_input("Product ID", "P001")

        with col2:
            st.subheader("💰 ราคาและโปรโมชั่น")
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

    # ===== 5. ส่วนการประมวลผล =====
    if submit_button:
        # จัดเตรียมข้อมูลให้ตรงกับ Column ที่ Pipeline ในโมเดลต้องการ (Case-sensitive)
        input_data = pd.DataFrame([{
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
        }])

        try:
            # ทำนายผล
            prediction = model.predict(input_data)[0]

            # แสดงผลลัพธ์
            st.balloons()
            st.markdown("---")
            st.subheader("📝 ผลการวิเคราะห์")
            
            # ตรวจสอบว่าค่าติดลบหรือไม่ (ป้องกันกรณีโมเดลทำนายเพี้ยน)
            final_sales = max(0, prediction)
            
            st.metric(label="ยอดขายที่คาดการณ์ (Predicted Sales)", value=f"{final_sales:,.2f} Units/Baht")
            
            # Progress bar จำลองระดับยอดขายเทียบกับเป้า 10,000
            st.write("**ระดับความหนาแน่นของยอดขายเมื่อเทียบกับเป้าหมาย:**")
            st.progress(min(final_sales / 10000, 1.0))

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")
            st.warning("ตรวจสอบว่าชื่อ Column ใน input_data ตรงกับที่โมเดลถูก Train มาหรือไม่")

else:
    st.warning("⚠️ ไม่สามารถเริ่มต้นระบบได้เนื่องจากโหลดโมเดลไม่สำเร็จ")

# ===== 6. คำแนะนำเพิ่มเติม =====
with st.expander("ℹ️ ข้อมูลทางเทคนิค (Technical Info)"):
    st.write(f"- **Scikit-learn version:** {sklearn.__version__}")
    st.write("- **Model Type:** Scikit-learn Pipeline (Preprocessing + Estimator)")
    st.write("- **Requirement:** ต้องการข้อมูลครบทั้ง 13 Features ตามที่ระบุใน Form")
