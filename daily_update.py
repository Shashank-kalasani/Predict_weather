import requests
import pandas as pd
import numpy as np
import joblib
import keras
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

API_KEY = "b8425717b72c4c4292d42808261301"
CITY = "Nagpur"

def fetch_live_weather():
    url = "https://api.weatherapi.com/v1/current.json"
    params = {"key": API_KEY, "q": CITY}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()  
        data = r.json()

        temp = data["current"]["temp_c"]
        time = pd.to_datetime(data["current"]["last_updated"]).floor("H")

        return time, temp

    except requests.exceptions.RequestException as e:
        print("⚠️ WeatherAPI request failed:", e)
        return None, None


def update_data():
    df=pd.read_csv(r"C:\Users\kalas\Desktop\desktopcopy\coding\ML_Projects\predict_weather\data\live_weather.csv")

    time,temp=fetch_live_weather()
    
    if time is None:
        print("❌ Skipping update (API unavailable)")
        return

    df["time"] = pd.to_datetime(df["time"])


    new_row={"time":time.floor("H"),"temp":temp}

    new_row['time']=pd.to_datetime(new_row['time'])

    df=pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)

    df= df.drop_duplicates(subset="time", keep="last")
    df= df.sort_values("time").reset_index(drop=True)
    
    df.to_csv(r'C:\Users\kalas\Desktop\desktopcopy\coding\ML_Projects\predict_weather\data\live_weather.csv',index=False)

if __name__ == "__main__":
    update_data()