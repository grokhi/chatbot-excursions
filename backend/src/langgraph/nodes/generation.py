from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from src.core.logger import logger
from src.handlers import llm_handler
from src.handlers.llm import LLMHandler


def generate(state: MessagesState):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """

    logger.info("GENERATE")

    messages = state["messages"]
    question = [m for m in messages if isinstance(m, HumanMessage)][-1].content
    last_message = messages[-1]

    docs = last_message.content

    prompt = ChatPromptTemplate(
        [
            (
                "system",
                (
                    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. Do not mention that you have used the provided context. "
                    "Use three sentences maximum and keep the answer concise.\n"
                    "Question: {question}"
                    "\nContext: {context} "
                    "\nAnswer:"
                ),
            )
        ]
    )

    # LLM
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    # llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, streaming=True)
    llm = llm_handler.llm
    response = llm.invoke(prompt.format(question=question, context=docs))

    return {"messages": [response]}
