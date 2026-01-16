import requests
import pandas as pd
import os
import pytz

API_KEY = os.getenv("WEATHER_API_KEY")
CITY = os.getenv("CITY", "Nagpur")

DATA_PATH = "data/live_weather.csv"
IST = pytz.timezone("Asia/Kolkata")


def normalize_time(series):
    times = pd.to_datetime(series, errors="coerce")
    if times.dt.tz is None:
        return times.dt.tz_localize(IST)
    else:
        return times.dt.tz_convert(IST)


def fetch_live_weather():
    url = "https://api.weatherapi.com/v1/current.json"
    params = {"key": API_KEY, "q": CITY}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    temp = data["current"]["temp_c"]
    time_ist = (
        pd.to_datetime(data["current"]["last_updated"])
        .tz_localize(IST)
        .floor("H")
    )

    return time_ist, temp


def update_data():
    df = pd.read_csv(DATA_PATH)
    df["time"] = normalize_time(df["time"])

    time, temp = fetch_live_weather()

    new_row = pd.DataFrame([{"time": time, "temp": temp}])
    df = pd.concat([df, new_row], ignore_index=True)

    df = df.drop_duplicates(subset="time", keep="last")
    df = df.sort_values("time").reset_index(drop=True)

    df.to_csv(DATA_PATH, index=False)
    print(f"Weather updated: {time} → {temp}°C")


if __name__ == "__main__":
    update_data()
