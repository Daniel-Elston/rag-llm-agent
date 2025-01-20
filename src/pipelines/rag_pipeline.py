from __future__ import annotations

from config.pipeline_context import PipelineContext
from utils.execution import TaskExecutor
from src.data.data_module import DataModule

from config.settings import Config, Params

from src.models.rag_build import RAGBuilder
from src.models.rag_gen import RAGGenerator
from src.models.hf_llm import LLMPipeline


class RAGPipeline:
    """
    Summary: Accesses FAISS vector store, builds RAG pipeline and generates responses.\n
    Input: FAISS store ``data_state key: faiss_store``\n
    Output: LLM generated response ``reports/outputs/generated-answers.txt``
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
            state_key="rag_pipeline",
        )
        
        self.local_llm = LLMPipeline(
            ctx=self.ctx,
        )

    def __call__(self):
        steps = [
            RAGBuilder(
                ctx = self.ctx,
                dm = self.dm_faiss,
                llm = self.local_llm,
            ),
            RAGGenerator(
                ctx = self.ctx,
                dm = self.dm_rag,
            )
        ]
        self.exe._execute_steps(steps, stage="parent")