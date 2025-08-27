import os
from typing import Optional, Tuple
import speech_recognition as sr
from gtts import gTTS
import requests
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableParallel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import tool, AgentExecutor
from langdetect import detect, LangDetectException

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
        
        # --- THIS IS THE CHANGE ---
        # 1. Extract more of the useful data points.
        location = data['location']['name']
        temp = data['current']['temp_c']
        condition = data['current']['condition']['text']
        wind_kph = data['current']['wind_kph']
        humidity = data['current']['humidity']
        
        # 2. Format them into a clean, multi-line summary.
        return (
            f"Current weather in {location}:\n"
            f"- Temperature: {temp}°C\n"
            f"- Condition: {condition}\n"
            f"- Wind Speed: {wind_kph} kph\n"
            f"- Humidity: {humidity}%"
        )
    else:
        return "Sorry, I couldn't fetch the weather data for that city."

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
