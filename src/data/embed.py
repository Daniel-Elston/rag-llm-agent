from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config.paths import Paths
from config.pipeline_context import PipelineContext
from config.settings import Config
from config.states import DataState
from src.data.data_module import DataModule
from utils.file_access import FileAccess


class BuildVectorStore:
    """
    Summary: Build a FAISS vector store from_documents
    Brief:
        - Automatically calls `self.embeddings.embed_documents` on each doc
        - Returns FAISS vector store
        - Stores FAISS vector store in state
    """

    def __init__(
        self, ctx: PipelineContext,
        dm: DataModule,
        embeddings: HuggingFaceEmbeddings
    ):
        self.ctx = ctx
        self.chunked_docs = dm.load()
        self.embeddings = embeddings
        self.config: Config = ctx.settings.config
        self.data_state: DataState = ctx.states.data
        self.paths: Paths = ctx.paths

    def __call__(self):
        faiss_store = FAISS.from_documents(
            self.chunked_docs,
            self.embeddings
        )
        self._save_helper(faiss_store, self.chunked_docs)

    def _log_doc_embeddings(self, faiss_store, chunked_docs):
        raw_vectors = faiss_store.index.reconstruct_n(0, len(chunked_docs))
        for i, vec in enumerate(raw_vectors[:2]):
            log_entry = (
                f"Embedding {i}, length{len(vec)}:\n{vec}\n\n"
            )
            FileAccess.save_file(
                log_entry,
                self.paths.get_path("embeddings_sample")
            )

    def _save_helper(self, faiss_store, chunked_docs):
        self.data_state.set("faiss_store", faiss_store)
        if self.config.write_output:
            self._log_doc_embeddings(faiss_store, chunked_docs)
