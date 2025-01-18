from __future__ import annotations

import logging

from config.state_init import StateManager
from src.data.data_module import DataModule

from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkText:
    """
    Summary: 
    Brief:
        - Standardise special characters if complex fonts
        - Remove excessive whitespace
        - Fix repeated punctuation (e.g. ". ." -> ".")
    """
    def __init__(
        self, state: StateManager,
        dm: DataModule,
        text_splitter: RecursiveCharacterTextSplitter
    ):
        self.state = state
        self.dm = dm
        self.text_splitter = text_splitter


    def __call__(self):
        documents = self.dm.load()
        chunks = self.text_splitter.split_documents(documents)
        chunks_filtered = [doc for doc in chunks if self.filter_junk_chunks(doc)]
        # self._log_doc_chunks(chunks_filtered)
        self.state.data_state.set("chunk_docs_all", chunks_filtered)
        return chunks_filtered

    def filter_junk_chunks(self, doc):
        """If chunk will is mostly useless numerical data, filter out"""
        content = doc.page_content
        alpha_chars = sum(char.isalpha() for char in content)
        return alpha_chars >= len(content) // 2
    
    def _log_doc_chunks(self, chunks_filtered):
        for i in (0, 1, 2, 15, -1):
            logging.debug(f"[Document {i} of {len(chunks_filtered)}]")
            logging.debug(chunks_filtered[i].page_content)
            logging.debug("="*125)