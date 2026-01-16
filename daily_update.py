import requests
import pandas as pd
import os
import pytz
from datetime import datetime

API_KEY = os.getenv("WEATHER_API_KEY")
CITY = os.getenv("CITY", "Nagpur")

DATA_PATH = "data/live_weather.csv"
IST = pytz.timezone("Asia/Kolkata")


def fetch_live_weather():
    url = "https://api.weatherapi.com/v1/current.json"
    params = {"key": API_KEY, "q": CITY}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    temp = data["current"]["temp_c"]
    time_ist = pd.to_datetime(
        data["current"]["last_updated"]
    ).tz_localize(IST)

    return time_ist, temp, data["current"]["last_updated"]


def update_data():
    print("=== DAILY UPDATE RUN ===")
    print("Execution time (IST):", datetime.now(IST))

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(IST)

    time, temp, api_time = fetch_live_weather()

    new_row = pd.DataFrame([{
        "time": time,
        "temp": temp,
        "api_last_updated": api_time
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df = df.drop_duplicates(subset="time", keep="last")
    df = df.sort_values("time").reset_index(drop=True)

    df.to_csv(DATA_PATH, index=False)
    print(f"Weather updated → {time} | {temp}°C")


if __name__ == "__main__":
    update_data()
