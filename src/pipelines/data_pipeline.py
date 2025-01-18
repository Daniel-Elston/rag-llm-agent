from __future__ import annotations

from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import PyPDFLoader


import logging
from pprint import pprint
from config.state_init import StateManager
from utils.execution import TaskExecutor
from src.data.data_module import DataModule, load_dataset
from src.data.process import ProcessChunks
from src.data.doc_loader import DocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )

    def __call__(self):
        DocumentLoader(self.state).run(),
        steps = [
            ProcessChunks(
                config = self.config,
                documents = load_dataset(self.dm_raw_docs),
                text_splitter = self.text_splitter
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
