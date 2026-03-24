import streamlit as st
import pandas as pd
import joblib
import sklearn

# ===== การตั้งค่าหน้าเว็บ =====
st.set_page_config(page_title="Retail Prediction", page_icon="📊")

# แสดงเวอร์ชันปัจจุบันเพื่อการตรวจสอบ (Debug)
st.sidebar.write(f"⚙️ System Check:")
st.sidebar.write(f"- Sklearn version: {sklearn.__version__}")

@st.cache_resource
def load_retail_model():
    try:
        # พยายามโหลดโมเดล
        return joblib.load("retail_sales_model.pkl")
    except Exception as e:
        return e

# ส่วนแสดงผลหน้าเว็บ
st.title("📊 ระบบทำนายยอดขายรายสินค้า")
st.divider()

model = load_retail_model()

# ตรวจสอบ Error
if isinstance(model, Exception):
    st.error(f"❌ Error: {model}")
    if "AttributeError" in str(model):
        st.warning("⚠️ สาเหตุ: เวอร์ชัน scikit-learn ไม่ตรงกับที่ใช้สร้างโมเดล")
        st.info("💡 วิธีแก้: ตรวจสอบว่าในเครื่องติดตั้ง scikit-learn==1.6.1 หรือยัง")
    st.stop() # หยุดการทำงานหากโหลดโมเดลไม่ได้

# --- ถ้าโหลดผ่านแล้ว จะแสดงฟอร์มด้านล่าง ---
st.success("✅ โหลดโมเดลเวอร์ชัน 1.6.1 สำเร็จ!")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        inventory = st.number_input("Inventory Level", value=500)
        units_ordered = st.number_input("Units Ordered", value=50)
        demand = st.number_input("Demand Forecast", value=100)
        price = st.number_input("Price", value=250.0)
        disc = st.slider("Discount (%)", 0, 100, 10)
        comp = st.number_input("Competitor Pricing", value=245.0)
        promo = st.selectbox("Holiday Promotion", [0, 1])

    with col2:
        sid = st.text_input("Store ID", "S001")
        pid = st.text_input("Product ID", "P001")
        cat = st.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Health & Beauty", "Toys"])
        reg = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        wea = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy"])
        sea = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

    if st.form_submit_button("🔮 ทำนายยอดขาย"):
        data = pd.DataFrame([{
            "Inventory_Level": inventory, "Units_Ordered": units_ordered,
            "Demand_Forecast": demand, "Price": price, "Discount": disc,
            "Competitor_Pricing": comp, "Holiday_Promotion": promo,
            "Store_ID": sid, "Product_ID": pid, "Category": cat,
            "Region": reg, "Weather_Condition": wea, "Seasonality": sea
        }])
        
        prediction = model.predict(data)[0]
        st.success(f"### ผลการพยากรณ์: {max(0, prediction):,.2f}")
