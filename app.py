import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== 1. ตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="ระบบพยากรณ์ยอดขายสินค้า",
    page_icon="📊",
    layout="centered"
)

# ===== 2. โหลดโมเดล =====
@st.cache_resource
def load_model():
    try:
        model = joblib.load("retail_sales_model.pkl")
        return model
    except Exception as e:
        return None, str(e)

model_result = load_model()

# ===== Header =====
st.title("📊 Retail Sales Prediction")
st.markdown("ระบบวิเคราะห์และพยากรณ์ยอดขายด้วย AI")
st.divider()

# ===== ตรวจสอบโมเดล =====
if isinstance(model_result, tuple):
    st.error(f"❌ โหลดโมเดลไม่ได้: {model_result[1]}")
else:
    model = model_result

    # ===== 3. Input =====
    with st.form("my_form"):
        st.subheader("📝 กรอกข้อมูลเพื่อพยากรณ์")

        col1, col2 = st.columns(2)

        with col1:
            inventory = st.number_input("Inventory Level", value=500)
            units_ordered = st.number_input("Units Ordered", value=50)
            demand_forecast = st.number_input("Demand Forecast", value=100)
            price = st.number_input("Price", value=250.0)
            discount = st.slider("Discount (%)", 0, 100, 10)
            comp_pricing = st.number_input("Competitor Pricing", value=245.0)
            holiday_promo = st.selectbox("Holiday Promotion", [0, 1])

        with col2:
            store_id = st.text_input("Store ID", "S001")
            product_id = st.text_input("Product ID", "P001")
            category = st.selectbox("Category", ["Electronics", "Clothing", "Home & Kitchen", "Health & Beauty", "Toys"])
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
            weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy"])
            season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

        submit = st.form_submit_button("🔮 ทำนายยอดขาย")

    # ===== 4. Predict =====
    if submit:
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
            # ===== สำคัญ: ต้องมี preprocessing ใน model =====
            prediction = model.predict(input_df)

            # กันค่าติดลบ
            prediction_value = float(max(0, prediction[0]))

            st.markdown("---")
            st.success("✅ วิเคราะห์เสร็จสิ้น")

            st.metric(
                label="ยอดขายที่คาดการณ์",
                value=f"{prediction_value:,.2f}"
            )

            with st.expander("🔍 ดูข้อมูลที่ใช้ทำนาย"):
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")

# ===== Footer =====
st.caption("Developed with ❤️ using Streamlit & Scikit-learn")
