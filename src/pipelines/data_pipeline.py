from __future__ import annotations

from config.pipeline_context import PipelineContext
from utils.execution import TaskExecutor
from src.data.data_module import DataModule

from config.settings import Params

from src.data.doc_loader import DocumentLoader
from src.data.process import ProcessDocuments

from langchain.text_splitter import RecursiveCharacterTextSplitter


class DataPipeline:
    """
    Summary: Loads all raw documents and performs light NLP cleaning.\n
        Input: Raw documents (local or arxiv) ``data_state key: raw_docs_all``\n
        Output: Lightly processed documents ``data_state key: proc_docs_all``
    """
    def __init__(
        self, ctx: PipelineContext,
        exe: TaskExecutor
    ):
        self.ctx = ctx
        self.exe = exe
        self.params: Params = ctx.settings.params
        
        self.dm_raw_docs = DataModule(
            ctx=self.ctx,
            state_key="raw_docs_all",
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.params.chunk_size,
            chunk_overlap=self.params.chunk_overlap,
            separators=self.params.separators,
        )


    def prepare_raw_data(self):
        DocumentLoader(ctx = self.ctx).run(),
        steps = [
            ProcessDocuments(
                ctx = self.ctx,
                dm = self.dm_raw_docs,
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
