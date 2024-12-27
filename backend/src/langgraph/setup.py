import sqlite3

import langchain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from src.core.config import config
from src.langgraph.edges.grading import grade_documents
from src.langgraph.nodes import agent, generate, retriever_tool, rewrite, web_search

try:  # docker
    db_path = "state_db/example.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
except:  # debug
    memory = MemorySaver()


if config.LANGCHAIN_DEBUG:
    langchain.debug = True


class LangGraphSetup:
    """Setup class for creating and managing the LangGraph workflow."""

    def create_workflow(self) -> StateGraph:
        """
        Create and configure the workflow workflow.

        Returns:
            StateGraph: Compiled workflow graph
        """
        workflow = StateGraph(MessagesState)

        workflow.add_node("agent", agent)
        workflow.add_node("retrieve", ToolNode([retriever_tool]))
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("generate", generate)
        # workflow.add_node("web_search", web_search)

        workflow.add_edge(START, "agent")
        # workflow.add_edge("agent", "retrieve")

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "retrieve",
            grade_documents,
        )
        workflow.add_edge("rewrite", "agent")
        # workflow.add_edge("web_search", END)
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=memory)


def create_graph() -> StateGraph:
    """
    Factory function to create and return a configured LangGraph instance.

    Returns:
        StateGraph: Configured processing graph
    """
    setup = LangGraphSetup()
    return setup.create_workflow()
