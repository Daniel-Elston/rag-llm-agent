from __future__ import annotations

import logging

from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import PyPDFLoader

from config.state_init import StateManager
from src.data.data_dict import NoDataDict


class DocumentLoader:
    def __init__(
        self, state: StateManager,
    ):
        self.state = state
        self.config = state.data_config

    def run(self):
        self.all_docs = []
        self.load_pdfs()
        self.load_arxiv()
        self.state.data_state.set("raw_docs_all", self.all_docs)

    def load_pdfs(self):
        pdf_path_idx = ["raw-t1", "raw-t2"]
        for idx in pdf_path_idx:
            pdf_path = self.state.paths.get_path(idx)
            loader = PyPDFLoader(pdf_path)
            self.all_docs.extend(loader.load())
        # self._log_doc_metadata(self.all_docs)
    
    def load_arxiv(self):
        arxiv_path_idx = ["raw-idx1", "raw-idx2"]
        for idx in arxiv_path_idx:
            arx_path = self.state.paths.get_path(idx)
            loader = ArxivLoader(query=str(arx_path))
            self.all_docs.extend(loader.load())
        # self._log_doc_metadata(self.all_docs)

    def _log_doc_metadata(self, documents):
        for i, doc in enumerate(documents):
            logging.debug(f"Document {i + 1}:")
            logging.debug(f"Page Content (Sample):\n{doc.page_content[1000:1050]}")
            logging.debug(f"Metadata:\n{doc.metadata}")
