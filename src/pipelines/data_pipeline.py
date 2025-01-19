from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor

from src.data.data_module import DataModule
from src.data.process import ProcessDocuments
from src.data.doc_loader import DocumentLoader
from src.data.chunk import ChunkDocuments

from langchain.text_splitter import RecursiveCharacterTextSplitter


class DataPipeline:
    """
    Summary: Loads all raw documents and performs light NLP cleaning.\n
        Input: Raw documents (local or arxiv) ``data_state key: raw_docs_all``\n
        Output: Chunked documents ``data_state key: proc_docs_all``
    """
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe
        self.data_config = state.data_config
        
        self.dm_raw_docs = DataModule(
            state=self.state,
            state_key="raw_docs_all",
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.data_config.chunk_size,
            chunk_overlap=self.data_config.chunk_overlap,
            separators=self.data_config.separators,
        )


    def prepare_data(self):
        DocumentLoader(
            state = self.state,
            data_config = self.data_config
        ).run(),
        steps = [
            ProcessDocuments(
                state = self.state,
                dm = self.dm_raw_docs,
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
