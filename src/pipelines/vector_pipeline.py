from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor

from src.data.data_module import DataModule
from src.data.embed import BuildVectorStore

from langchain_huggingface import HuggingFaceEmbeddings


class VectorStorePipeline:
    """
    Summary: Takes chunked documents, performs embedding, and stores them in FAISS (in-memory).\n
        Input: Chunked documents ``data_state key: chunk_docs_all``\n
        Output: Embedded documents stored in FAISS ``data_state key: faiss_store``
    """
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe
        self.config = state.model_config
        
        self.dm_chunk_docs = DataModule(
            state=self.state,
            state_key="chunk_docs_all",
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name
        )


    def build_vector_store(self):
        steps = [
            BuildVectorStore(
                state = self.state,
                dm = self.dm_chunk_docs,
                embeddings = self.embeddings
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
