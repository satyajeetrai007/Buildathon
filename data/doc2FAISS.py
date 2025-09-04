import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

folder_path = "data/knowledge_base"   
loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

print(f"Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

print(f"Split into {len(docs)} chunks")

# Instead of HuggingFace Model, Now Using Gemini Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


vectorstore = FAISS.from_documents(docs, embedding_model)

db_path = "agri_faiss_index"
vectorstore.save_local(db_path)
print("✅ Vectorstore saved at", db_path)