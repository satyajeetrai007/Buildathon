import os
import textwrap
from typing import Optional, Tuple
import speech_recognition as sr
from gtts import gTTS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableParallel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langdetect import detect, LangDetectException

LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def ask_and_speak(query: str, audio_path: str, rag_chain: Runnable) -> None:
    try:
        detected_lang = detect(query)
    except LangDetectException:
        detected_lang = "en"

    ai_message_response = rag_chain.invoke(query)
    text_answer = ai_message_response.content

    if not text_answer:
        return

    wrapped_text = textwrap.fill(text_answer, width=100)
    print(f"Generated Text Answer:\n{wrapped_text}")

    tts_obj = gTTS(text=text_answer, lang=detected_lang)
    output_audio_path = "answer.mp3"
    tts_obj.save(output_audio_path)
    print(f"SUCCESS! Audio answer saved to {output_audio_path}")


def setup_hybrid_rag_chain(llm, search_tool, vectorstore_retriever) -> Runnable:
    """
    Hybrid RAG chain:
    - Uses embeddings (query → vector similarity search in FAISS)
    - Uses raw query for web search
    """
    template = """
    You are a helpful assistant. Answer the user's question based on the following sources:
    
    - Context from stored knowledge (books, PDFs, reports, etc.)
    - Context from trusted web search results
    
    Provide a clear, concise, synthesized answer in the same language as the question.  
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
        print("Listening... Please ask your question.")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        with open(audio_file_path, "wb") as f:
            f.write(audio.get_wav_data())

        query = recognizer.recognize_google(audio)
        print(f"User said: {query}")
        return query, audio_file_path


def ask_text(query: str, rag_chain: Runnable) -> str:
    """
    Handles pure textual queries.
    Detects the language of the query,
    queries the RAG chain,
    and returns the answer in the same language.
    """
    try:
        detected_lang = detect(query)
    except LangDetectException:
        detected_lang = "en"

    # Run through RAG chain
    ai_message_response = rag_chain.invoke(query)
    text_answer = ai_message_response.content if hasattr(ai_message_response, "content") else str(ai_message_response)

    if not text_answer:
        return "I don’t know."

    wrapped_text = textwrap.fill(text_answer, width=100)
    print(f"Generated Text Answer:\n{wrapped_text}")

    return text_answer
