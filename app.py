import streamlit as st
import pandas as pd
import pickle

def drop_columns(X):
    return X.drop(
        columns=['Age', 'Store Rating', 'Discount (%)', 'Loyalty Score'],
        errors='ignore'
    )

# Load model

@st.cache_resource
def load_model():
    with open("models/rf_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("📈 Revenue Prediction App")


# Sidebar Inputs

st.sidebar.header("Customer Details")

items_purchased = st.sidebar.number_input(
    "Items Purchased", min_value=1, step=1
)

total_spent = st.sidebar.number_input(
    "Total Spent", min_value=0.0, step=100.0
)

satisfaction_score = st.sidebar.slider(
    "Satisfaction Score", min_value=1.0, max_value=5.0, step=0.1
)

warranty_extension = st.sidebar.selectbox(
    "Warranty Extension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)

membership_status = st.sidebar.selectbox(
    "Membership Status", [0, 1], format_func=lambda x: "Member" if x == 1 else "Non-member"
)

gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female","Other"]
)

region = st.sidebar.selectbox(
    "Region", ["North", "South", "East", "West"]
)

product_category = st.sidebar.selectbox(
    "Product Category", ["Accessories", "Laptop", "Mobile", "Tablet","Television"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method", ["Credit Card", "Debit Card", "UPI", "Cash","Net Banking"]
)

preferred_visit_time = st.sidebar.selectbox(
    "Preferred Visit Time", ["Morning", "Afternoon", "Evening"]
)


# Create Input DataFrame

input_data = pd.DataFrame({
    "Items Purchased": [items_purchased],
    "Total Spent": [total_spent],
    "Satisfaction Score": [satisfaction_score],
    "Warranty Extension": [warranty_extension],
    "Membership Status": [membership_status],
    "Gender": [gender],
    "Region": [region],
    "Product Category": [product_category],
    "Payment Method": [payment_method],
    "Preferred Visit Time": [preferred_visit_time]
})

st.subheader("Input Data")
st.write(input_data)

# Prediction

if st.button("Predict Revenue"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Revenue: {prediction:,.2f}")
