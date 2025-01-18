from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor

from src.data.data_module import DataModule
from src.data.process import ProcessText
from src.data.doc_loader import DocumentLoader
from src.data.chunk import ChunkText

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import PyPDFLoader


class DataPipeline:
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe
        self.config = state.data_config
        
        self.dm_raw_docs = DataModule(
            state=self.state,
            state_key="raw_docs_all",
        )
        self.dm_processed_docs = DataModule(
            state=self.state,
            state_key="proc_docs_all",
        )
        self.dm_chunk_docs = DataModule(
            state=self.state,
            state_key="chunk_docs_all",
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )


    def __call__(self):
        DocumentLoader(self.state).run(),
        steps = [
            ProcessText(
                state = self.state,
                dm = self.dm_raw_docs,
            ),
            ChunkText(
                state = self.state,
                dm = self.dm_processed_docs,
                text_splitter = self.text_splitter,
            ),
        ]
        self.exe._execute_steps(steps, stage="parent")
