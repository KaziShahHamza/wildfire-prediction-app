import requests

LAT, LON = 36.7783, -119.4179  # California center
API_KEY = "f1e7e631ad639459f0e21c004841161b"  # Replace with your API key

def get_weather():
    """Fetch live weather data for California."""
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    r = requests.get(url).json()
    precipitation = r.get("rain", {}).get("1h", 0.0)
    max_temp = r["main"]["temp_max"]
    min_temp = r["main"]["temp_min"]
    wind_speed = r["wind"]["speed"]
    return precipitation, max_temp, min_temp, wind_speed
