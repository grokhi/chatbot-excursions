import hashlib
import json
import os
from typing import Any, List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.embeddings import Embeddings
from src.core.config import config


class VectorDBHandler:

    _instance = None
    VECTORSTORE_DIR = "vectorstore_cache"
    DS_HASH_DIR = "dataset_hash"

    def __init__(self, collection_name: str, embedding_model: Embeddings):
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._vectorstore = None
        self._retriever = None
        self._cached_file_hash = None
        self._force_reload = None

        self._vectorstore_path = os.path.join(
            config.DATA_DIR, self.VECTORSTORE_DIR, collection_name
        )
        os.makedirs(self._vectorstore_path, exist_ok=True)

        ds_hash_dirpath = os.path.join(config.DATA_DIR, self.DS_HASH_DIR)
        os.makedirs(ds_hash_dirpath, exist_ok=True)
        self._ds_hash_path = os.path.join(ds_hash_dirpath, collection_name)

    def load_data_from_json(self, file_path: str, force_reload: bool = False) -> List[Document]:
        """
        Load data from a JSON file and cache it for subsequent use.
        Reload if the file content changes or force_reload is True.

        Args:
            file_path (str): Path to the JSON file.
            force_reload (bool): Force reloading of the document data. Defaults to False.

        Returns:
            list[Document]: List of processed LangChain Document objects.
        """
        file_hash = self._compute_file_hash(file_path)
        self._cached_file_hash = self._load_hash()
        self._force_reload = self._cached_file_hash != file_hash or force_reload

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        documents = []
        for item in data:
            content = (
                f"Question (RU): {item['question_text']}\n"
                f"Question (EN): {item.get('question_eng', '')}\n"
                f"Answer: {item['answer_text']}\n"
            )
            metadata = {
                "uid": item["uid"],
                "tags": item["tags"],
                "version": item["RuBQ_version"],
            }
            documents.append(Document(page_content=content, metadata=metadata))

        self._save_hash(file_hash)

        return documents

    def create_vectorstore(
        self, documents: List[Document], chunk_size: int = 250, chunk_overlap: int = 0
    ):
        """
        Split documents into chunks and create a Chroma vectorstore.

        Args:
            documents (list[Document]): List of LangChain Document objects.
            chunk_size (int): Size of text chunks. Defaults to 250.
            chunk_overlap (int): Overlap between text chunks. Defaults to 0.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        filter_complex_metadata(documents)
        splitted_docs = text_splitter.split_documents(documents)

        # Check if vectorstore cache exists
        # if os.path.exists(self._vectorstore_path) and self._force_reload is False:
        #     self._vectorstore = Chroma(
        #         persist_directory=self._vectorstore_path,
        #         collection_name=self._collection_name,
        #         embedding_function=self._embedding_model,
        #     )
        # else:
        #     self._vectorstore = Chroma.from_documents(
        #         documents=splitted_docs,
        #         collection_name=self._collection_name,
        #         embedding=self._embedding_model,
        #         persist_directory=self._vectorstore_path,
        #     )

        self._vectorstore = Chroma.from_documents(
            documents=splitted_docs,
            collection_name=self._collection_name,
            embedding=self._embedding_model,
        )
        return self._vectorstore

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """
        Compute a hash for the given file to detect changes.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: Hash of the file content.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _save_hash(self, hash_value: str):
        """
        Save the hash to a file.

        Args:
            hash_value (str): Hash value to save.
        """
        with open(self._ds_hash_path, "w") as hash_file:
            hash_file.write(hash_value)

    def _load_hash(self) -> Optional[str]:
        """
        Load the hash from a file.

        Returns:
            Optional[str]: The loaded hash value or None if the file doesn't exist.
        """
        if os.path.exists(self._ds_hash_path):
            with open(self._ds_hash_path, "r") as hash_file:
                return hash_file.read()
        return None
