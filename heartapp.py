import streamlit as st
import pandas as pd
import joblib
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# SIMPLE LOGIN SYSTEM
# -----------------------------
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "12345":
            st.session_state["login"] = True
        else:
            st.error("Invalid username or password")

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
    st.stop()

# -----------------------------
# LOAD DATA AND MODEL
# -----------------------------
df=pd.read_excel("Heart/Book1.xlsx")
lr = pd.read_csv("Heart/heart.csv")
X = lr.drop("target", axis=1)

model = joblib.load("linear.pkl")
scaler = joblib.load("scale3.pkl")

# -----------------------------
# APP HEADER
# -----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write(
"""
This application predicts **heart disease risk** using machine learning.

Fill the patient information in the sidebar.
"""
)

# -----------------------------
# USER INPUT
# -----------------------------
st.sidebar.header("Patient Information")

user_input = {}

for col in X.columns:

    if X[col].dtype in ["int64", "float64"]:

        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())

        if min_val == max_val:
            user_input[col] = min_val
        else:
            user_input[col] = st.sidebar.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    else:
        user_input[col] = st.sidebar.selectbox(col, X[col].unique())

user_df = pd.DataFrame([user_input])

# -----------------------------
# DISPLAY USER INPUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("User Input Data")
    st.dataframe(user_df)

with col2:
    st.subheader("Input Visualization")
    st.bar_chart(user_df.T)

# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("Prediction")

if st.button("Predict Heart Disease Risk"):

    user_scaled = scaler.transform(user_df)

    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")

# -----------------------------
# RESET BUTTON
# -----------------------------
if st.button("Reset App"):
    st.rerun()
