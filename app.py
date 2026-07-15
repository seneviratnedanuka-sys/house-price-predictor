import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

model = load_model()

st.title("🏠 House Price Predictor")
st.write("Estimate a sale price from a property's key characteristics.")

col1, col2 = st.columns(2)
with col1:
    sqft = st.number_input("Living area (sqft)", 400, 6000, 1800, step=50)
    bedrooms = st.slider("Bedrooms", 1, 7, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    lot_size = st.number_input("Lot size (sqft)", 500, 45000, 5000, step=100)
    garage_spaces = st.selectbox("Garage spaces", [0, 1, 2, 3], index=2)
with col2:
    year_built = st.number_input("Year built", 1900, 2026, 1995)
    dist_to_center_km = st.number_input("Distance to city centre (km)", 0.1, 40.0, 6.0)
    quality_score = st.slider("Overall quality / condition (1-10)", 1.0, 10.0, 6.5)
    renovated = st.selectbox("Recently renovated?", ["No", "Yes"]) == "Yes"
    neighborhood = st.selectbox(
        "Neighborhood", ["Riverside", "Oakwood", "Downtown", "Hillcrest", "Maple Grove"]
    )

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "lot_size": lot_size, "year_built": year_built, "garage_spaces": garage_spaces,
        "dist_to_center_km": dist_to_center_km, "quality_score": quality_score,
        "renovated": int(renovated), "neighborhood": neighborhood,
    }])
    pred = model.predict(row)[0]
    st.success(f"### Estimated price: ${pred:,.0f}")
    st.caption("Estimate from a tuned Gradient Boosting model — for guidance only.")

st.divider()
st.caption("Model: Gradient Boosting Regressor · Test R² = 0.958 · Test MAE ≈ $20,200")
