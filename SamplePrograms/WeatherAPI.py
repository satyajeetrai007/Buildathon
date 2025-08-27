import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Your Details ---
# Get the API key from the environment variables
API_KEY = os.getenv("WEATHER_API_KEY") 
CITY = "Jaipur"

# --- Construct the URL and Make the Request ---
request_url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={CITY}"
response = requests.get(request_url)

# --- Check, Parse, and Print ---
if response.status_code == 200:
    data = response.json()
    # ... (the rest of the printing logic is the same)
    print(data)
else:
    print("Error fetching data.")