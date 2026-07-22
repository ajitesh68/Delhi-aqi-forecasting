import requests
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")



def get_lat_lon(city_name):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},in&limit=1&appid={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data:
            return data[0]['lat'], data[0]['lon']
        else:
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Geocoding API Error for {city_name}: {e}") 
        return None, None


def get_live_pollution(lat,lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and "list" in data and len(data["list"]) > 0:
            return data["list"][0]["components"]
        else:
            return None
        
    except requests.exceptions.RequestException as e: 
        print(f"Pollution API Error: {e}")
        return None 
        

    