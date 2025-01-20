from __future__ import annotations

from config.pipeline_context import PipelineContext
from utils.file_access import FileAccess

from config.paths import Paths
from config.settings import Config
from config.states import DataState

from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader:
    """
    Summary: Load all raw documents from local dir or arxiv
    """
    def __init__(
        self, ctx: PipelineContext
    ):
        self.ctx = ctx
        self.config: Config = ctx.settings.config
        self.data_state: DataState = ctx.states.data
        self.paths: Paths = ctx.paths
        
    def run(self):
        self.all_docs = []
        self.load_pdfs()
        self.load_arxiv()
        self._save_helper()

    def load_pdfs(self):
        # pdf_path_idx = ["raw-t1", "raw-t2"]
        pdf_path_idx = ["raw-p1", "raw-p2"]
        for idx in pdf_path_idx:
            pdf_path = self.paths.get_path(idx)
            loader = PyPDFLoader(pdf_path)
            self.all_docs.extend(loader.load())
    
    def load_arxiv(self):
        arxiv_path_idx = ["raw-idx1", "raw-idx2"]
        for idx in arxiv_path_idx:
            arx_path = self.paths.get_path(idx)
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
                self.paths.get_path("raw-doc-metadata")
            )
    
    def _save_helper(self):
        self.data_state.set("raw_docs_all", self.all_docs)
        if self.config.write_output:
            self._log_doc_metadata(self.all_docs)