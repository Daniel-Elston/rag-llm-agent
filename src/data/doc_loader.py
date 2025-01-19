from __future__ import annotations

from utils.file_access import FileAccess
from config.data import DataConfig

from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import PyPDFLoader

from config.state_init import StateManager


class DocumentLoader:
    """
    Summary: Load all raw documents from local dir or arxiv
    """
    def __init__(
        self, state: StateManager,
        data_config: DataConfig
    ):
        self.state = state
        self.data_config = data_config

    def run(self):
        # self.state.data_state.clear()
        self.all_docs = []
        self.load_pdfs()
        self.load_arxiv()
        self._save_helper()
        # self.state.data_state.set("raw_docs_all", self.all_docs)
        # if self.config.write_output:
        #     self._log_doc_metadata(self.all_docs)

    def load_pdfs(self):
        # pdf_path_idx = ["raw-t1", "raw-t2"]
        pdf_path_idx = ["raw-p1", "raw-p2"]
        for idx in pdf_path_idx:
            pdf_path = self.state.paths.get_path(idx)
            loader = PyPDFLoader(pdf_path)
            self.all_docs.extend(loader.load())
    
    def load_arxiv(self):
        arxiv_path_idx = ["raw-idx1", "raw-idx2"]
        for idx in arxiv_path_idx:
            arx_path = self.state.paths.get_path(idx)
            loader = ArxivLoader(query=str(arx_path))
            self.all_docs.extend(loader.load())
    
    def _log_doc_metadata(self, documents):
        for i, doc in enumerate(documents):
            log_entry = (
                f"Document {i + 1}:\n"
                f"Metadata:\n{doc.metadata}\n"
                f"Page Content (Sample):\n{doc.page_content[250:1000]}\n\n"
            )
            FileAccess.save_file(
                log_entry, 
                self.state.paths.get_path("raw-doc-metadata")
            )
    
    def _save_helper(self):
        self.state.data_state.set("raw_docs_all", self.all_docs)
        if self.data_config.write_output:
            self._log_doc_metadata(self.all_docs)