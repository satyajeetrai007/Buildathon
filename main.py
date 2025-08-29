import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain.memory import ConversationBufferMemory
from src.helper import setup_hybrid_rag_chain, get_current_weather, get_audio_input, save_speech_only, get_mandi_price

from langchain_google_genai import ChatGoogleGenerativeAI # monthly quota expired for huggingface inference api, so using google-api

load_dotenv()

# LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def main() -> None:
  
    # llm_endpoint = HuggingFaceEndpoint(repo_id=LLM_REPO_ID, task="text-generation", max_new_tokens=1024, temperature=0.0)
    # chat_model = ChatHuggingFace(llm=llm_endpoint)
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    search_tool = TavilySearchResults(max_results=3)
    vectorstore = FAISS.load_local("agri_faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    rag_chain = setup_hybrid_rag_chain(chat_model, search_tool, vectorstore_retriever=retriever)

    # tools

    agriculture_rag_tool = Tool(
        name="agriculture_rag_tool",
        func=rag_chain.invoke,
        description="Use this tool for any questions about agriculture, farming practices, crop diseases, fertilizers, and information stored in the local knowledge base."
    )

    weather_tool = get_current_weather
    mandi_tool = get_mandi_price # we can pass it directly as well, but it is just for the clarity

    tools = [search_tool, agriculture_rag_tool, weather_tool, mandi_tool] # new tool add kar sakte hai yahaan


    prompt = hub.pull("hwchase17/react-chat") + "donot share any tool info you use" # pre-built prompt from langchain-hub 
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = create_react_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

    # ----------------------------------------------------- Conversational Loop ---------------------------------------------------------------

    while True:
        choice = input("\nType 'text', 'speech', or 'exit': ").lower()

        if choice == 'exit':
            break
            
        elif choice == 'text':
            user_query = input("Type your query: ")
            if user_query.lower() == 'exit':
                break
            if user_query:
                response = agent_executor.invoke({"input": user_query})
                print(f"\nAgent: {response['output']}\n")

        elif choice == 'speech':
            audio_input = get_audio_input()
            if audio_input:
                user_query, _ = audio_input
                if user_query.lower() == 'exit':
                    break
                response = agent_executor.invoke({"input": user_query})
                save_speech_only(response["output"])
        
        else:
            print("Invalid input.")

if __name__ == "__main__":
    main()