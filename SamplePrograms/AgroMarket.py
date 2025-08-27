import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

def get_mandi_price(api_key: str, commodity: str, state: str, district: str, date: str) -> dict | None:
    """
    Fetches market price for a commodity on a specific date using the data.gov.in API.
    """
    base_url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 10,
        "filters[State]": state,
        "filters[District]": district,
        "filters[Commodity]": commodity,
        "filters[Arrival_Date]": date # Use the date passed to the function
    }

    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            print("SUCCESS: API request was successful.")
            return response.json()
        else:
            print(f"ERROR: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"CONNECTION ERROR: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    # 1. IMPORTANT: Replace this with your personal key from data.gov.in
    YOUR_PERSONAL_API_KEY = os.getenv("MARKET_PRICE_API_KEY")

    # 2. TEST with a past date to ensure data exists
    test_date = "25/08/2025" 

    test_commodity = "Onion"
    test_state = "Maharashtra"
    test_district = "Pune"
    
    print(f"Fetching price for {test_commodity} in {test_district}, {test_state} on {test_date}...")
    
    price_data = get_mandi_price(
        api_key=YOUR_PERSONAL_API_KEY,
        commodity=test_commodity,
        state=test_state,
        district=test_district,
        date=test_date # Pass the test date to the function
    )

    if price_data and 'records' in price_data and price_data['records']:
        print("\n--- Market Price Data ---")
        print(json.dumps(price_data['records'], indent=2))
    elif price_data and 'records' in price_data:
        print(f"\n--- No records found for {test_date}. Try an even earlier date. ---")
    else:
        print("\n--- Failed to retrieve data. ---")