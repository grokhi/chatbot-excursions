from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from src.core.logger import logger
from src.handlers import vectorstore

# retriever = vectorstore.as_retriever()
similarity_soft_threshold: float = 0.5


@tool("retrieve_docs")
def retriever_tool(query: str):
    "Search in vectorstore and return information relevant to the input query"
    retrieved_docs = vectorstore.similarity_search_with_score(query, k=5)
    # filtered = [d for d in retrieved_docs if d[1] < similarity_soft_threshold]
    # if len(filtered) == 0:
    #     return "irrelevant"
    return "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc, _ in retrieved_docs
    )


# retriever_tool = create_retriever_tool(
#     retriever,
#     "retrieve_documents",
#     "Search and return information relevenat to the input question",
# )
