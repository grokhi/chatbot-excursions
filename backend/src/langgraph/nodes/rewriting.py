from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from src.core.logger import logger
from src.handlers import llm_handler
from src.handlers.llm import LLMHandler


def rewrite(state: MessagesState):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    logger.info("TRANSFORM QUERY")

    messages = state["messages"]
    # question = messages[0].content
    question = [m for m in messages if isinstance(m, HumanMessage)][-1].content

    msg = [
        HumanMessage(
            content=f""" \n 
                Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                Here is the initial question:
                \n ------- \n
                {question} 
                \n ------- \n
                Formulate an improved question. Do not mention your reasoning. Be lapidary as possible.""",
        )
    ]

    # Grader
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    # model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)
    llm = llm_handler.llm

    response = llm.invoke(msg)
    return {"messages": [response]}
