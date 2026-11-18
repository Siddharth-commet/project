"""
app.py
Streamlit frontend: enter a new customer's details -> predicted CAC and CLV.
Run with: streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CLV & CAC Predictor", layout="centered")
st.title("New Customer: CLV & CAC Predictor")


@st.cache_resource
def load():
    return joblib.load("models.joblib")


b = load()
opt, d = b["options"], b["defaults"]

with st.form("customer"):
    st.subheader("Profile & acquisition")
    c1, c2 = st.columns(2)
    age = c1.number_input("Customer age", 18, 100, int(d["customer_age"]))
    gender = c2.selectbox("Gender", opt["gender"])
    segment = c1.selectbox("Customer segment", opt["customer_segment"])
    region = c2.selectbox("Region", opt["region"])
    mkt = c1.selectbox("Marketing channel", opt["marketing_channel"])
    sales = c2.selectbox("Sales channel", opt["sales_channel"])

    st.subheader("Behavior so far (for CLV)")
    st.caption("For a brand-new customer, use 0 / low values for orders and profit.")
    c3, c4 = st.columns(2)
    aov = c3.number_input("Avg order value", 0.0, value=float(d["avg_order_value"]))
    freq = c4.number_input("Purchase frequency (orders/month)", 0.0,
                           value=float(d["purchase_frequency"]), step=0.1)
    n_orders = c3.number_input("Total orders so far", 0, value=int(d["total_orders"]))
    profit = c4.number_input("Total profit so far", value=float(d["total_profit"]))
    recency = c3.number_input("Days since last order", 0, value=int(d["recency_days"]))
    go = st.form_submit_button("Predict")

if go:
    row = pd.DataFrame([{
        "customer_age": age, "gender": gender, "customer_segment": segment,
        "region": region, "marketing_channel": mkt, "sales_channel": sales,
        "avg_order_value": aov, "purchase_frequency": freq,
        "total_orders": n_orders, "total_profit": profit, "recency_days": recency,
    }])

    cac = max(float(b["cac_model"].predict(row[b["cac_features"]])[0]), 0)
    clv = max(float(b["clv_model"].predict(row[b["clv_features"]])[0]), 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted CAC", f"{cac:,.2f}")
    m2.metric("Predicted 12-month CLV", f"{clv:,.2f}")
    ratio = clv / cac if cac > 0 else None
    m3.metric("CLV / CAC", f"{ratio:.2f}" if ratio is not None else "n/a")

    if ratio is not None:
        if ratio >= 3:
            st.success("CLV:CAC of 3+ is the usual healthy benchmark.")
        elif ratio >= 1:
            st.warning("Profitable but thin margin.")
        else:
            st.error("This customer is predicted to cost more than they return.")

    st.caption(f"Models used: CAC = {b['cac_name']}, CLV = {b['clv_name']}")