from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor

from src.data.data_module import DataModule
from src.data.chunk import ChunkDocuments

from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkPipeline:
    """
    Summary: Takes processed documents, performs chunking.\n
        Input: Processed documents ``data_state key: proc_docs_all``\n
        Output: Chunked documents ``data_state key: chunk_docs_all``
    """
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe
        self.config = state.data_config
        
        self.dm_processed_docs = DataModule(
            state=self.state,
            state_key="proc_docs_all",
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )


    def __call__(self):
        steps = [
            ChunkDocuments(
                state = self.state,
                dm = self.dm_processed_docs,
                text_splitter = self.text_splitter,
            ),
        ]
        self.exe._execute_steps(steps, stage="parent")
