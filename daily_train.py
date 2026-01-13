import pandas as pd
import numpy as np
import joblib
import keras
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def create_sequences(series,window):
    X,y=[],[]
    for i in range(window,len(series)):
        X.append(series[i-window:i])
        y.append(series[i,0])
    return np.array(X),np.array(y)

window=24
epochs_daily=2


def train_model():
    df=pd.read_csv(r'C:\Users\kalas\Desktop\desktopcopy\coding\ML_Projects\predict_weather\data\live_weather.csv')

    scaler=joblib.load(r'C:\Users\kalas\Desktop\desktopcopy\coding\ML_Projects\predict_weather\model\temp_scaler.pkl')

    model = load_model(
        r'C:\Users\kalas\Desktop\desktopcopy\coding\ML_Projects\predict_weather\model\temp_lstm.h5',
        compile=False
    )

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='mse'
    )

    scaled=scaler.transform(df[['temp']].values)

    X,y=create_sequences(scaled,window)

    model.fit(X,y,epochs=epochs_daily,batch_size=32,verbose=1)

    model.save(r'C:\Users\kalas\Desktop\desktopcopy\coding\ML_Projects\predict_weather\model\temp_lstm.h5')

if __name__ == "__main__":
    train_model()