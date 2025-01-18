from __future__ import annotations

import logging
from utils.file_access import FileAccess
from config.state_init import StateManager
from src.data.data_module import DataModule

from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkDocuments:
    """
    Summary: Splits documents into chunks, based on text splitter
    Brief:
        - Splits documents into chunks using text splitter and data config params
        - Filters out chunks with mostly numerical data
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
        """Logs and saves specific chunks of documents to a file"""
        log_entries = []
        for i in (0, 1, 2, 15, -1):
            if i < len(chunks_filtered):  # Avoid out-of-range errors
                log_entries.append(
                    f"[Document {i} of {len(chunks_filtered)}]\n"
                    f"Chunk:\n{chunks_filtered[i].page_content}\n\n"
                    f"{'=' * 125}\n"
                )
        combined_log = "\n".join(log_entries)
        FileAccess.save_file(
                    combined_log, self.state.paths.get_path("sample-chunks"))
