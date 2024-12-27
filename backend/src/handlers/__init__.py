from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from src.core.config import config
from src.handlers.llm import LLMHandler
from src.handlers.vector_db import VectorDBHandler

# initialize handlers
llm_handler = LLMHandler(run_local=config.RUN_LOCAL_LLM)
vector_handler = VectorDBHandler(
    collection_name="qa_phuket",
    embedding_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    # embedding_model=OpenAIEmbeddings(),
)
# documents = vector_handler.load_data_from_json("data/datasets/RuBQ_2.0_dev.json")
documents = vector_handler.load_data_from_json("data/datasets/excursions.json")
vectorstore = vector_handler.create_vectorstore(documents)
