import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import altair as alt
from datetime import datetime
import pytz

# ---------------- CONFIG ----------------
DATA_PATH = "data/live_weather.csv"
MODEL_PATH = "model/temp_lstm.keras"
SCALER_PATH = "model/temp_scaler.pkl"
WINDOW = 24

IST = pytz.timezone("Asia/Kolkata")

st.set_page_config(
    page_title="Weather Prediction",
    layout="wide"
)

# ---------------- LOAD DATA (NO CACHE) ----------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    times = pd.to_datetime(df["time"], errors="coerce")

    if times.dt.tz is None:
        df["time"] = times.dt.tz_localize(IST)
    else:
        df["time"] = times.dt.tz_convert(IST)

    df = df.sort_values("time").reset_index(drop=True)
    return df



def load_model_and_scaler():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


df = load_data()
model, scaler = load_model_and_scaler()

# ---------------- HEADER ----------------
st.title("🌦️ Weather Forecast (IST)")
st.caption("All times shown in IST (Asia/Kolkata)")

last_time = df["time"].iloc[-1]
st.markdown(
    f"**Last available data:** {last_time.strftime('%d %b %Y, %I:%M %p')} IST"
)

# ---------------- DEBUG: APP RESTART TIME ----------------
st.sidebar.success(
    f"App restarted at:\n{datetime.now(IST).strftime('%d %b %Y %I:%M:%S %p IST')}"
)

# ---------------- LAST 48 HOURS ----------------
st.subheader("📉 Last 48 Hours Temperature")

hist_df = df.tail(48)

hist_chart = (
    alt.Chart(hist_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "time:T",
            title="Time (IST)",
            axis=alt.Axis(format="%d %b %I:%M %p")
        ),
        y=alt.Y("temp:Q", title="Temperature (°C)"),
        tooltip=[
            alt.Tooltip("time:T", title="Time (IST)", format="%d %b %I:%M %p"),
            alt.Tooltip("temp:Q", title="Temperature (°C)")
        ]
    )
    .properties(height=350)
)

st.altair_chart(hist_chart, use_container_width=True)

# ---------------- NEXT 24 HOURS FORECAST ----------------
st.subheader("🔮 Next 24 Hours Forecast")

temps = df["temp"].values.reshape(-1, 1)
scaled = scaler.transform(temps)

last_seq = scaled[-WINDOW:]
last_seq = last_seq.reshape(1, WINDOW, 1)

preds = []
current_seq = last_seq.copy()

for _ in range(24):
    pred = model.predict(current_seq, verbose=0)[0, 0]
    preds.append(pred)
    current_seq = np.append(
        current_seq[:, 1:, :],
        [[[pred]]],
        axis=1
    )

preds = scaler.inverse_transform(
    np.array(preds).reshape(-1, 1)
).flatten()

future_times = pd.date_range(
    start=last_time + pd.Timedelta(hours=1),
    periods=24,
    freq="H",
    tz=IST
)

pred_df = pd.DataFrame({
    "time": future_times,
    "pred_temp": preds
})

# ---------------- FORECAST CHART ----------------
pred_chart = (
    alt.Chart(pred_df)
    .mark_line(point=True, color="orange")
    .encode(
        x=alt.X(
            "time:T",
            title="Time (IST)",
            axis=alt.Axis(format="%d %b %I:%M %p")
        ),
        y=alt.Y("pred_temp:Q", title="Predicted Temperature (°C)"),
        tooltip=[
            alt.Tooltip("time:T", title="Time (IST)", format="%d %b %I:%M %p"),
            alt.Tooltip("pred_temp:Q", title="Temperature (°C)")
        ]
    )
    .properties(height=350)
)

st.altair_chart(pred_chart, use_container_width=True)

# ---------------- FORECAST TABLE ----------------
st.subheader("📋 Next 24 Hours – Table (IST)")

table_df = pred_df.copy()
table_df["Time (IST)"] = table_df["time"].dt.strftime("%d %b %I:%M %p")
table_df["Temperature (°C)"] = table_df["pred_temp"].round(1)

st.dataframe(
    table_df[["Time (IST)", "Temperature (°C)"]],
    use_container_width=True,
    hide_index=True
)

# ---------------- FOOTER ----------------
now_ist = datetime.now(IST)

st.caption(
    f"⏱ Current IST time: {now_ist.strftime('%d %b %Y, %I:%M %p')} | "
    "Hourly updates via GitHub Actions"
)
