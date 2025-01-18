from __future__ import annotations

import logging
import numpy as np
from utils.file_access import FileAccess
from config.state_init import StateManager
from src.data.data_module import DataModule

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class BuildVectorStore:
    """
    Summary: Build a FAISS vector store from_documents
    Brief:
        - Automatically calls `self.embeddings.embed_documents` on each doc
        - Returns FAISS vector store
        - Stores FAISS vector store in state
    """
    def __init__(
        self, state: StateManager,
        dm: DataModule,
        embeddings: HuggingFaceEmbeddings
    ):
        self.state = state
        self.dm = dm
        self.embeddings = embeddings

    def __call__(self):
        chunked_docs = self.state.data_state.get("chunk_docs_all")
        faiss_store = FAISS.from_documents(
            chunked_docs,
            self.embeddings
        )
        # self._log_doc_embeddings(faiss_store, chunked_docs)
        self.state.data_state.set("faiss_store", faiss_store)
    
    def _log_doc_embeddings(self, faiss_store, chunked_docs):
        raw_vectors = faiss_store.index.reconstruct_n(0, len(chunked_docs))
        for i, vec in enumerate(raw_vectors[:2]):  # first 2 embeddings
            log_entry = (
                f"Embedding {i}, length{len(vec)}:\n{vec}\n\n"
            )
            FileAccess.save_file(
                log_entry, self.state.paths.get_path("embeddings_sample")
            )
