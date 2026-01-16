import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pytz

DATA_PATH = "data/live_weather.csv"
MODEL_PATH = "model/temp_lstm.keras"
SCALER_PATH = "model/temp_scaler.pkl"

WINDOW = 24          # last 24 OBSERVATIONS (not hours)
EPOCHS_DAILY = 2
IST = pytz.timezone("Asia/Kolkata")


def normalize_time(series):
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize(IST)
    return ts.dt.tz_convert(IST)


def create_sequences(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def train_model():
    print("=== DAILY TRAIN STARTED ===")

    df = pd.read_csv(DATA_PATH)
    df["time"] = normalize_time(df["time"])
    df = df.sort_values("time")

    temps = df[["temp"]].values

    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(temps)

    X, y = create_sequences(scaled, WINDOW)

    model = load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    model.fit(
        X,
        y,
        epochs=EPOCHS_DAILY,
        batch_size=32,
        verbose=1
    )

    model.save(MODEL_PATH)

    print("✅ Daily training completed")


if __name__ == "__main__":
    train_model()
