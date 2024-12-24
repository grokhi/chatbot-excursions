from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from src.core.logger import logger
from src.handlers import llm_handler
from src.langgraph.nodes.retrieving import retriever_tool


def agent(state: MessagesState):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the human question, the agent should always decide to choose retriever tool.
    The query given to the retriever tool should consider context of previous human messages.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """

    logger.info("CALL AGENT")

    messages = state["messages"]
    # model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    # model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)
    llm = llm_handler.llm

    llm_with_tools = llm.bind_tools([retriever_tool], tool_choice="retrieve_docs")
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
