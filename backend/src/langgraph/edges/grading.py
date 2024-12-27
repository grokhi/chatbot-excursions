from typing import Annotated, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, MessagesState
from pydantic import BaseModel, Field
from src.core.logger import logger
from src.handlers.llm import LLMHandler


def grade_documents(state: MessagesState) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    logger.info("CHECK RELEVANCE")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # LLM with tool and validation
    model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)
    # llm = LLMHandler().llm
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    docs = last_message.content

    if docs == "irrelevant":
        logger.info("DECISION: DOCS NOT RELEVANT")
        return "web_search"

    try:
        msg = [m for m in messages if isinstance(m, AIMessage)][-1]
        question = msg.tool_calls[0]["args"]["query"]
    except:
        msg = [m for m in messages if isinstance(m, HumanMessage)][-1]
        question = msg.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        logger.info("DECISION: DOCS RELEVANT")
        return "generate"

    else:
        logger.info("DECISION: DOCS NOT RELEVANT")
        # return "web_search"
        return "rewrite"
