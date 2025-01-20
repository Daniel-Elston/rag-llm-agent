from __future__ import annotations

from config.pipeline_context import PipelineContext
from utils.execution import TaskExecutor
from src.data.data_module import DataModule

from config.settings import Config, Params

from src.data.rag_build import RAGBuilder
from src.data.rag_gen import RAGGenerator


class RAGPipeline:
    """
    Summary: Takes chunked documents, performs embedding, and stores them in FAISS (in-memory).\n
        Input: Chunked documents ``data_state key: chunk_docs_all``\n
        Output: Embedded documents stored in FAISS ``data_state key: faiss_store``
    """
    def __init__(
        self, ctx: PipelineContext,
        exe: TaskExecutor
    ):
        self.ctx = ctx
        self.exe = exe
        self.config: Config = ctx.settings.config
        self.params: Params = ctx.settings.params
        
        self.dm_faiss = DataModule(
            ctx=self.ctx,
            state_key="faiss_store",
        )
        self.dm_rag = DataModule(
            ctx=self.ctx,
            state_key="qa_chain",
        )

    def __call__(self):
        steps = [
            RAGBuilder(
                ctx = self.ctx,
                dm = self.dm_faiss,
            ),
            RAGGenerator(
                ctx = self.ctx,
                dm = self.dm_rag,
            )
        ]
        self.exe._execute_steps(steps, stage="parent")