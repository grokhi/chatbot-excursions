from typing import Any, Dict, Literal

from langchain_community.llms.gpt4all import GPT4All
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.core.config import config
from src.core.logger import logger


class LLMHandler:

    def __init__(self, run_local: bool = True):

        if run_local:
            self.llm = ChatOllama(
                model=config.LOCAL_MODEL,
                temperature=0,
                streaming=True,
                base_url=config.LOCAL_LLM_HOST,
            )
        else:
            self.llm = ChatGroq(
                model=config.API_MODEL,
                temperature=0,
                streaming=True,
            )
            # self.llm = ChatOpenAI(
            #     model=config.OPENAI_MODEL,
            #     temperature=0,
            #     streaming=True,
            # )
