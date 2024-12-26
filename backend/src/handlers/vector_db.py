import hashlib
import json
import os
from typing import Any, List, Optional

import requests
from bs4 import BeautifulSoup
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

    def parse_excursion_document(self, item: dict) -> Document:
        """Parse excursion data into LangChain Document format"""

        # Create content string
        content = (
            f"Title: {item['title']}\n"
            f"Description: {item['description']}\n"
            f"About: {item['tour_about']}\n"
        )

        # Format prices for metadata
        formatted_prices = {}
        for price_group in item["prices"]:
            if not price_group.get("prices"):
                continue

            program_prices = {
                price["type"]: price["price"]
                for price in price_group["prices"]
                if "type" in price and "price" in price
            }

            if program_prices:
                formatted_prices[price_group["title"]] = program_prices

        # Create metadata
        metadata = {"title": item["title"], "prices": str(formatted_prices)}

        return Document(page_content=content, metadata=metadata)

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
        # file_hash = self._compute_file_hash(file_path)
        # self._cached_file_hash = self._load_hash()
        # self._force_reload = self._cached_file_hash != file_hash or force_reload

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        documents = []
        for item in data:
            doc = self.parse_excursion_document(item)
            documents.append(doc)

        # self._save_hash(file_hash)

        return documents

    def load_data_from_website(self, url: str, force_reload: bool = False) -> List[Document]:
        """
        Load data from a website and cache it for subsequent use.
        Reload if the content changes or force_reload is True.

        Args:
            url (str): URL of the website.
            force_reload (bool): Force reloading of the document data. Defaults to False.

        Returns:
            list[Document]: List of processed LangChain Document objects.
        """
        response = requests.get(url)
        response.raise_for_status()
        # content_hash = hashlib.sha256(response.content).hexdigest()
        # self._cached_file_hash = self._load_hash()
        # self._force_reload = self._cached_file_hash != content_hash or force_reload

        soup = BeautifulSoup(response.content, "html.parser")
        excursions = self._parse_excursions(soup)

        documents = []
        for excursion in excursions:
            content = (
                f"Title: {excursion['title']}\n"
                f"Description: {excursion['description']}\n"
                f"Price: {excursion['price']}\n"
            )
            metadata = {
                "url": url,
                "tags": excursion.get("tags", []),
            }
            documents.append(Document(page_content=content, metadata=metadata))

        # self._save_hash(content_hash)

        return documents

    def _get_tour_description(self, soup):
        """Extract description from tour-about section"""
        tour_about = soup.find("section", class_="tour-about")
        if not tour_about:
            return ""

        # Get title
        title = tour_about.find("h2", class_="title")
        title_text = title.get_text(strip=True) if title else ""

        # Get content paragraphs
        paragraphs = tour_about.select(".tour-about__column p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        return f"{title_text}. {content}"

    def _parse_prices(self, soup):
        """Parse prices from program list items"""
        prices = []

        # Find all program items
        programs = soup.select(".program-list__item")

        for program in programs:
            price_data = {"title": "", "prices": []}

            # Get program title
            title = program.select_one(".program-list__title")
            if title:
                price_data["title"] = title.get_text(strip=True)

            # Get prices for each tariff
            tariffs = program.select(".program-list__tarif")
            for tariff in tariffs:
                tariff_data = {}

                # Get tariff name (Adult/Child)
                name = tariff.select_one(".program-list__tarif-name")
                if name:
                    tariff_data["type"] = name.get_text(strip=True)

                # Get price
                price = tariff.select_one(".program-list__tarif-price")
                if price:
                    # Clean price text (remove '฿' and 'from')
                    price_text = price.get_text(strip=True)
                    price_text = price_text.replace("฿", "").replace("from ", "")
                    tariff_data["price"] = int(price_text)

                if tariff_data:
                    price_data["prices"].append(tariff_data)

            prices.append(price_data)

        return prices

    def _parse_excursions(self, soup: BeautifulSoup) -> List[dict]:
        """
        Parse excursion information from the BeautifulSoup object.

        Args:
            soup (BeautifulSoup): BeautifulSoup object containing the website content.

        Returns:
            list[dict]: List of dictionaries with excursion information.
        """
        excursions = []
        for excursion_div in soup.select(" .nav__link"):

            link = excursion_div.get("href")
            if (
                not link
                or not link.startswith("https://phuket-cheap-tour.com/")
                or link.endswith("/catalog")
            ):
                continue

            title = excursion_div.get_text(strip=True)

            response = requests.get(link)
            if response.status_code != 200:
                continue

            excursion_soup = BeautifulSoup(response.content, "html.parser")

            try:
                prices = self._parse_prices(excursion_soup)
            except Exception as e:
                continue

            og_desc = excursion_soup.find("meta", property="og:description")
            if og_desc:
                description = og_desc.get("content")

            tour_about = self._get_tour_description(excursion_soup)

            excursions.append(
                {
                    "title": title,
                    "description": description,
                    "tour_about": tour_about,
                    "prices": prices,
                }
            )
        return excursions

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
