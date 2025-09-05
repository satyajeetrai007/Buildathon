import os
import json
import requests
from dotenv import load_dotenv

# We import @tool so you can paste your function directly without modification
from langchain.agents import tool

# --- Paste the function you want to test here ---
@tool
def get_mandi_price(tool_input: str | dict) -> str:
    """
    Fetches the daily market (mandi) price for an agricultural commodity.
    The input to this tool MUST be a JSON dictionary with the keys 'commodity', 
    'state', 'district', and 'date'. The date MUST be in DD/MM/YYYY format.
    """
    try:
        if isinstance(tool_input, dict):
            input_dict = tool_input
        else:
            clean_str = tool_input.strip().lstrip("```json").rstrip("```").strip()
            input_dict = json.loads(clean_str)
    except (json.JSONDecodeError, AttributeError):
        return "Error: The tool received a malformed input. Please ensure the Action Input is a valid JSON dictionary."

    commodity = input_dict.get("commodity")
    state = input_dict.get("state")
    district = input_dict.get("district")
    date = input_dict.get("date")

    if not all([commodity, state, district, date]):
        return "Error: The input dictionary is missing required keys. It must contain 'commodity', 'state', 'district', and 'date'."

    # Corrected environment variable name to match previous setup
    api_key = os.getenv("DATA_GOV_API_KEY") 
    if not api_key:
        return "ERROR: The data.gov.in API key is not configured."
    
    base_url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    params = {
        "api-key": api_key, "format": "json", "limit": 5,
        "filters[State]": state, "filters[District]": district,
        "filters[Commodity]": commodity, "filters[Arrival_Date]": date
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data and 'records' in data and data['records']:
                records = data['records']
                summary = f"Found {len(records)} price records for {commodity} in {district} on {date}:\n"
                for record in records:
                    summary += (
                        f"- Market: {record.get('Market', 'N/A')}, "
                        f"Variety: {record.get('Variety', 'N/A')}, "
                        f"Min Price: {record.get('Min_x0020_Price', 'N/A')}, "
                        f"Max Price: {record.get('Max_x0020_Price', 'N/A')}\n"
                    )
                return summary
            else:
                return f"No market price data found for {commodity} in {district} on {date}."
        else:
            return f"Error: API returned status code {response.status_code}. Response: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the API: {e}"

# --- This block will run when you execute the script ---
if __name__ == "__main__":
    load_dotenv()
    print("--- Starting Tool Test ---")
    
    # Define a sample input to test with
    test_date = "28/08/2025"

    # Test Case 1: Input is a clean JSON string (like the agent should provide)
    print("\n[TEST 1: Clean JSON string]")
    json_string_input = f'{{"commodity": "Onion", "state": "Maharashtra", "district": "Pune", "date": "{test_date}"}}'
    result1 = get_mandi_price(json_string_input)
    print("Result:\n", result1)
    print("-" * 20)

    # Test Case 2: Input is a "messy" string with markdown
    print("\n[TEST 2: Messy string with markdown backticks]")
    messy_string_input = f'`{{"commodity": "Onion", "state": "Maharashtra", "district": "Pune", "date": "{test_date}"}}`'
    result2 = get_mandi_price(messy_string_input)
    print("Result:\n", result2)
    print("-" * 20)

    # Test Case 3: Input is an invalid string
    print("\n[TEST 3: Invalid string that is not JSON]")
    invalid_string_input = "commodity=Onion, state=Maharashtra"
    result3 = get_mandi_price(invalid_string_input)
    print("Result:\n", result3)
    print("-" * 20)