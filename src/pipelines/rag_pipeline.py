from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor

from src.data.data_module import DataModule

from src.data.rag_build import RAGBuilder
from src.data.rag_gen import RAGGenerator


class RAGPipeline:
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
        self.data_config = state.data_config
        self.model_config = state.model_config
        
        self.dm_faiss = DataModule(
            state=self.state,
            state_key="faiss_store",
        )
        self.dm_rag = DataModule(
            state=self.state,
            state_key="qa_chain",
        )

    def __call__(self):
        steps = [
            RAGBuilder(
                state = self.state,
                model_config = self.model_config,
                dm = self.dm_faiss,
            ),
            RAGGenerator(
                state = self.state,
                data_config = self.data_config,
                model_config = self.model_config,
                dm = self.dm_rag,
            )
        ]
        self.exe._execute_steps(steps, stage="parent")