from __future__ import annotations

from config.pipeline_context import PipelineContext
from utils.execution import TaskExecutor
from src.data.data_module import DataModule

from config.settings import Config, Params

from src.data.chunk import ChunkDocuments

from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkPipeline:
    """
    Summary: Takes processed documents, performs chunking.\n
        Input: Processed documents ``data_state key: proc_docs_all``\n
        Output: Chunked documents ``data_state key: chunk_docs_all``
    """
    def __init__(
        self, ctx: PipelineContext,
        exe: TaskExecutor
    ):
        self.ctx = ctx
        self.exe = exe
        self.params: Params = ctx.settings.params
        self.config: Config = ctx.settings.config
        
        self.dm_processed_docs = DataModule(
            ctx=self.ctx,
            state_key="proc_docs_all",
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.params.chunk_size,
            chunk_overlap=self.params.chunk_overlap,
            separators=self.params.separators,
        )


    def chunk_data(self):
        steps = [
            ChunkDocuments(
                ctx = self.ctx,
                dm = self.dm_processed_docs,
                text_splitter = self.text_splitter,
            ),
        ]
        self.exe._execute_steps(steps, stage="parent")
