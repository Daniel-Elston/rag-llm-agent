from __future__ import annotations

from config.data import DataConfig
from src.data.data_module import DataModule

import logging
import re
from pprint import pprint
import unicodedata
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ProcessChunks:
    """
    Summary: Light NLP cleaning for document text
    Brief:
        - Standardise special characters if complex fonts
        - Remove excessive whitespace
        - Fix repeated punctuation (e.g. ". ." -> ".")
    """
    def __init__(
        self, config: DataConfig,
        documents: list,
        text_splitter: RecursiveCharacterTextSplitter
    ):
        self.config = config
        self.documents = documents
        self.text_splitter = text_splitter

    def __call__(self):
        for doc in self.documents:
            doc.page_content = self.clean_document_text(doc.page_content)
        chunks = self.text_splitter.split_documents(self.documents)
        chunks_filtered = [doc for doc in chunks if self.filter_junk_chunks(doc)]
        self.view(chunks_filtered)
        # doc.page_content = "\n".join(chunks)
        return chunks_filtered

    def clean_document_text(self, text: str) -> str:
        """Light NLP cleaning for document text."""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\.\s*\.", ".", text)
        # Remove reference bracket [1], [2]...? Could be useful
        # text = re.sub(r"\[\d+\]", "", text)
        return text

    def filter_junk_chunks(self, doc):
        """If chunk will is mostly useless numerical data, filter out"""
        content = doc.page_content
        alpha_chars = sum(char.isalpha() for char in content)
        return alpha_chars >= len(content) // 2
    
    def view(self, chunks_filtered):
        for i in (0, 1, 2, 15, -1):
            logging.debug(f"[Document {i} of {len(chunks_filtered)}]")
            logging.debug(chunks_filtered[i].page_content)
            logging.debug("="*125)