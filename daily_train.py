import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

DATA_PATH = "data/live_weather.csv"
MODEL_PATH = "model/temp_lstm.keras"
SCALER_PATH = "model/temp_scaler.pkl"

WINDOW = 24
EPOCHS_DAILY = 2


def create_sequences(series, window):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i, 0])
    return np.array(X), np.array(y)


def train_model():
    df = pd.read_csv(DATA_PATH)

    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH, compile=False)

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")

    scaled = scaler.transform(df[["temp"]].values)

    X, y = create_sequences(scaled, WINDOW)

    model.fit(
        X,
        y,
        epochs=EPOCHS_DAILY,
        batch_size=32,
        verbose=1
    )

    # ✅ SAVE BACK TO THE SAME FILE (RELATIVE PATH)
    model.save(MODEL_PATH)

    print("✅ Model updated and saved")


if __name__ == "__main__":
    train_model()
