import uuid
from typing import Any, Dict, List, Optional

from fastapi import Cookie, FastAPI, HTTPException, Response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from src.core.config import config
from src.core.logger import logger
from src.langgraph.setup import create_graph

app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG-based chatbot with document retrieval and LLM integration",
    version="1.0.0",
)

graph = create_graph()


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str = Field(default="What is the weather in sf?")
    # session_id: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    # messages: list


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, session: str = Cookie(None)) -> Dict[str, Any]:
    """
    Process chat queries using RAG approach

    Args:
        request: QueryRequest containing query and optional context

    Returns:
        Dict containing answer, sources, and confidence score
    """
    try:
        logger.info(f"Received query: {request.query}")
        config = {"configurable": {"thread_id": session}}
        res = graph.invoke(
            {
                "messages": [HumanMessage(content=request.query)],
            },
            config,
        )

        return {
            "question": request.query,
            "answer": res["messages"][-1].content,
        }

    except Exception as e:
        logger.exception(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}
