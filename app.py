import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

# ===== 1. ตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="ระบบพยากรณ์ยอดขายสินค้า",
    page_icon="📊",
    layout="centered"
)

# ===== 2. ฟังก์ชันโหลดโมเดล (ใช้ Cache เพื่อความเร็ว) =====
@st.cache_resource
def load_model():
    try:
        # โหลดโมเดลจากไฟล์ที่คุณเตรียมไว้
        return joblib.load("retail_sales_model.pkl")
    except Exception as e:
        return f"error: {str(e)}"

model_result = load_model()

# ส่วนหัวของเว็บ
st.title("📊 Retail Sales Prediction")
st.markdown("ระบบวิเคราะห์และพยากรณ์ยอดขายด้วย AI")
st.divider()

# ตรวจสอบว่าโหลดโมเดลสำเร็จไหม
if isinstance(model_result, str):
    st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {model_result}")
    st.info("💡 วิธีแก้: ตรวจสอบว่ามีไฟล์ 'retail_sales_model.pkl' อยู่ในโฟลเดอร์เดียวกับโค้ดนี้")
else:
    model = model_result
    
    # ===== 3. ส่วนรับข้อมูลจากผู้ใช้ (Input Form) =====
    with st.form("my_form"):
        st.subheader("📝 กรอกข้อมูลเพื่อพยากรณ์")
        
        col1, col2 = st.columns(2)
        
        with col1:
            inventory = st.number_input("Inventory Level (ระดับสินค้าคงคลัง)", value=500)
            units_ordered = st.number_input("Units Ordered (จำนวนที่สั่งซื้อ)", value=50)
            demand_forecast = st.number_input("Demand Forecast (คาดการณ์ความต้องการ)", value=100)
            price = st.number_input("Price (ราคาสินค้า)", value=250.0)
            discount = st.slider("Discount % (ส่วนลด)", 0, 100, 10)
            comp_pricing = st.number_input("Competitor Pricing (ราคาคู่แข่ง)", value=245.0)
            holiday_promo = st.selectbox("Holiday Promotion", [0, 1], format_func=lambda x: "มีโปรโมชั่น" if x == 1 else "ไม่มี")

        with col2:
            store_id = st.text_input("Store ID", "S001")
            product_id = st.text_input("Product ID", "P001")
            category = st.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Health & Beauty", "Toys"])
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
            weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Stormy"])
            season = st.selectbox("Seasonality", ["Spring", "Summer", "Autumn", "Winter"])

        submit = st.form_submit_button("🔮 ทำนายยอดขาย")

    # ===== 4. ส่วนแสดงผลการทำนาย =====
    if submit:
        # จัดเตรียมข้อมูลให้ตรงกับ Column ของ Model (สำคัญมาก!)
        input_df = pd.DataFrame([{
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
            prediction = model.predict(input_df)[0]
            
            st.markdown("---")
            st.success("### วิเคราะห์เสร็จสิ้น")
            
            # ตกแต่งการแสดงผลตัวเลข
            st.metric(label="ยอดขายที่คาดการณ์ได้", value=f"{max(0, prediction):,.2f} หน่วย/บาท")
            
            # แสดงรายละเอียดข้อมูลที่ใช้ทำนาย
            with st.expander("🔍 ดูข้อมูลที่ส่งให้ AI"):
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างทำนาย: {e}")

# Footer
st.caption("Developed by Gemini | Scikit-learn 1.6.1 Ready")
