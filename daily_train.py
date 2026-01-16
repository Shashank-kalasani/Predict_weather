import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pytz

DATA_PATH = "data/live_weather.csv"
MODEL_PATH = "model/temp_lstm.keras"
SCALER_PATH = "model/temp_scaler.pkl"

WINDOW = 24
EPOCHS_DAILY = 2
IST = pytz.timezone("Asia/Kolkata")


def create_sequences(series, window):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i, 0])
    return np.array(X), np.array(y)


def train_model():
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(IST)

    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH, compile=False)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    scaled = scaler.transform(df[["temp"]].values)
    X, y = create_sequences(scaled, WINDOW)

    model.fit(X, y, epochs=EPOCHS_DAILY, batch_size=32, verbose=1)

    model.save(MODEL_PATH)

    print("✅ Daily training complete")


if __name__ == "__main__":
    train_model()
