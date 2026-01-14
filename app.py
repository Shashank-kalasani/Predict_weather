import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ================= CONFIG =================
DATA_PATH = "data/live_weather.csv"
MODEL_PATH = "model/temp_lstm.keras"
SCALER_PATH = "model/temp_scaler.pkl"
WINDOW = 24
# =========================================

st.set_page_config(
    page_title="Tomorrow Weather Forecast",
    layout="centered"
)

st.title("🌦️ Tomorrow Weather Prediction (Next 24 Hours)")

# -------- Load model & scaler (cached) --------
@st.cache_resource
def load_model_and_scaler():
    model = load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=Adam(0.001),
        loss="mse"
    )
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# -------- Load data (cached) --------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

model, scaler = load_model_and_scaler()
df = load_data()

# -------- Prediction logic --------
def predict_next_24h(df):
    last_time = df["time"].iloc[-1]

    # last 24 hours (scaled)
    temps = scaler.transform(df[["temp"]].values)[-WINDOW:]

    preds_scaled = []

    for _ in range(24):
        x = temps.reshape(1, WINDOW, 1)
        p = model.predict(x, verbose=0)[0][0]
        preds_scaled.append(p)
        temps = np.vstack([temps[1:], [[p]]])

    preds = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=24,
        freq="h"
    )

    forecast_df = pd.DataFrame({
        "time": future_times,
        "predicted_temp": preds
    })

    return forecast_df

# ================= UI =================

st.subheader("📊 Recent Temperature (Last 48 Hours)")
st.line_chart(
    df.set_index("time")["temp"].tail(48)
)

st.caption(f"Last available data: **{df['time'].iloc[-1]}**")

st.divider()

if st.button("🔮 Predict Tomorrow (Next 24 Hours)"):
    forecast_df = predict_next_24h(df)

    st.subheader("🌡️ Hourly Forecast (Next 24 Hours)")

    # ✅ Proper datetime index → correct tooltip
    st.line_chart(
        forecast_df.set_index("time")["predicted_temp"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🌞 Max Temp",
        f"{forecast_df['predicted_temp'].max():.2f} °C"
    )
    col2.metric(
        "🌤️ Avg Temp",
        f"{forecast_df['predicted_temp'].mean():.2f} °C"
    )
    col3.metric(
        "🌙 Min Temp",
        f"{forecast_df['predicted_temp'].min():.2f} °C"
    )

    st.subheader("📋 Detailed Forecast Table")
    st.dataframe(
        forecast_df.style.format({
            "predicted_temp": "{:.2f} °C"
        }),
        use_container_width=True
    )

st.divider()

st.caption(
    "Model auto-updates on server • "
    "Forecast window moves with time • "
    "No manual retraining needed"
)
