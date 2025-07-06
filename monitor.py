# monitor.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="📈 Churn Model Monitor", layout="wide")
st.title("📊 Churn Model Monitoring Dashboard")

log_file = "prediction_logs.csv"

# --- Load logs ---
if not os.path.exists(log_file):
    st.warning("No predictions have been logged yet.")
    st.stop()

df = pd.read_csv(log_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Sidebar Filters ---
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", df["timestamp"].min().date())
end_date = st.sidebar.date_input("End Date", df["timestamp"].max().date())

filtered = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]

# --- Summary Metrics ---
st.subheader("📌 Summary Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", len(filtered))
col2.metric("Churn Predictions", filtered["prediction"].sum())
col3.metric("Churn Rate", f"{filtered['prediction'].mean() * 100:.2f}%" if len(filtered) > 0 else "N/A")

# --- Time Series Plot ---
st.subheader("📈 Churn Predictions Over Time")
time_series = filtered.copy()
time_series["date"] = time_series["timestamp"].dt.date
daily_counts = time_series.groupby("date")["prediction"].agg(["count", "sum"])
daily_counts["churn_rate"] = daily_counts["sum"] / daily_counts["count"]

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=daily_counts, x=daily_counts.index, y="churn_rate", ax=ax1)
ax1.set_title("Daily Churn Rate")
ax1.set_ylabel("Churn Probability")
ax1.set_xlabel("Date")
st.pyplot(fig1)

# --- Feature Distribution ---
st.subheader("🔍 Input Feature Monitoring")

# Categorical
categorical_cols = [col for col in filtered.columns if filtered[col].dtype == "object" and col not in ["timestamp", "prediction"]]
if categorical_cols:
    cat_col = st.selectbox("Select Categorical Feature", categorical_cols)
    st.bar_chart(filtered[cat_col].value_counts())

# Numerical
numerical_cols = filtered.select_dtypes(include=["float", "int"]).columns.drop(["prediction", "probability"], errors="ignore")
if len(numerical_cols) > 0:
    num_col = st.selectbox("Select Numeric Feature", numerical_cols)
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered[num_col], kde=True, ax=ax2)
    st.pyplot(fig2)
