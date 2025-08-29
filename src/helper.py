import os
from typing import Optional, Tuple
import speech_recognition as sr
from gtts import gTTS
import requests
from dotenv import load_dotenv
from langchain.agents import tool
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableParallel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import tool, AgentExecutor
from langdetect import detect, LangDetectException
import json
load_dotenv()

@tool
def get_current_weather(city: str) -> str:
    """
    Use this function to get the current real-time weather for a given city.
    Returns a string summarizing the temperature, weather conditions, wind speed, and humidity.
    """
    API_KEY = os.getenv("WEATHER_API_KEY")
    if not API_KEY:
        return "Weather API key is not configured."
        
    request_url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    response = requests.get(request_url)
    
    if response.status_code == 200:
        data = response.json()
        return data
        
        # # --- THIS IS THE CHANGE ---
        # # 1. Extract more of the useful data points.
        # location = data['location']['name']
        # temp = data['current']['temp_c']
        # condition = data['current']['condition']['text']
        # wind_kph = data['current']['wind_kph']
        # humidity = data['current']['humidity']
        
        # # 2. Format them into a clean, multi-line summary.
        # return (
        #     f"Current weather in {location}:\n"
        #     f"- Temperature: {temp}°C\n"
        #     f"- Condition: {condition}\n"
        #     f"- Wind Speed: {wind_kph} kph\n"
        #     f"- Humidity: {humidity}%"
        # )
    else:
        return "Sorry, I couldn't fetch the weather data for that city."


@tool
def get_mandi_price(tool_input: str | dict) -> str:
    """
    Fetches the daily market (mandi) price for an agricultural commodity.
    The input to this tool MUST be a JSON dictionary with the keys 'commodity', 
    'state', 'district', and 'date'. The date MUST be in DD/MM/YYYY format.
    """
    # --- NEW ROBUST PARSING LOGIC ---
    try:
        # If LangChain already parsed it into a dict, use it directly.
        if isinstance(tool_input, dict):
            input_dict = tool_input
        # If it's a string, we need to clean and parse it.
        else:
            # Clean the string of any markdown formatting (backticks, "json" markers)
            clean_str = tool_input.strip().lstrip("```json").rstrip("```").strip()
            input_dict = json.loads(clean_str)

    except (json.JSONDecodeError, AttributeError):
        return "Error: The tool received a malformed input. Please ensure the Action Input is a valid JSON dictionary."
    # --- END OF FIX ---

    # The rest of the function uses the parsed `input_dict`.
    commodity = input_dict.get("commodity")
    state = input_dict.get("state")
    district = input_dict.get("district")
    date = input_dict.get("date")

    if not all([commodity, state, district, date]):
        return "Error: The input dictionary is missing required keys. It must contain 'commodity', 'state', 'district', and 'date'."

    # The rest of the API call logic is exactly the same...
    api_key = os.getenv("MARKET_PRICE_API_KEY")
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

def setup_hybrid_rag_chain(llm, search_tool, vectorstore_retriever) -> Runnable:

    template = """
    You are a helpful assistant. Answer the user's question based on all available context.
    Provide a clear, synthesized answer in the same language as the question. 
    If you do not know the answer, just say: "I don’t know".

    Context from stored database:
    {db_context}

    Context from web search:
    {web_context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    combined_context = RunnableParallel(
        db_context=vectorstore_retriever,
        web_context=search_tool,
        question=RunnablePassthrough(),
    )
    return combined_context | prompt | llm



def get_audio_input() -> Optional[Tuple[str, str]]:

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0
    audio_file_path = "user_audio.wav"
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... Please ask your question or say 'exit'.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            with open(audio_file_path, "wb") as f:
                f.write(audio.get_wav_data())
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            return query, audio_file_path
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            print("Could not understand audio or no speech detected.")
            return None

def save_speech_only(text: str, lang: str = "en") -> None:
    """
    Converts text to speech and saves it as an MP3 file.
    """
    try:
        tts_obj = gTTS(text=text, lang=lang, slow=False)
        output_audio_path = "agent_response.mp3"
        tts_obj.save(output_audio_path)
        print(f"Agent: {text}")
        print(f"--> Spoken response saved to {output_audio_path}")
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
