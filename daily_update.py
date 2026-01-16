import requests
import pandas as pd
import os
import pytz

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

    # WeatherAPI time is LOCAL → force IST explicitly
    time_ist = pd.to_datetime(
        data["current"]["last_updated"]
    ).tz_localize(IST).floor("H")

    return time_ist, temp


def update_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(IST)

    time, temp = fetch_live_weather()

    new_row = pd.DataFrame(
        [{"time": time, "temp": temp}]
    )

    df = pd.concat([df, new_row], ignore_index=True)

    # Remove duplicates by hour
    df = df.drop_duplicates(subset="time", keep="last")
    df =
