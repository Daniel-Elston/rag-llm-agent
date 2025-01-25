from __future__ import annotations

from config.pipeline_context import PipelineContext
from src.data.data_module import DataModule
from src.models.hf_llm import LLMGenerator
from src.models.response_generator import RAGResponseGenerator
from src.models.retrieval_builder import RAGRetrievalBuilder
from utils.execution import TaskExecutor


class RAGRetrievalPipeline:
    """
    Summary: Builds the retrieval system and generates responses for single-turn QA.
        Combines document retrieval and LLM-based response generation.\n
        Input: FAISS vector store ``data_state key: faiss_store``\n
        Output: Generated responses ``data_state key: generated_answers``
    """

    def __init__(
        self, ctx: PipelineContext,
        exe: TaskExecutor
    ):
        self.ctx = ctx
        self.exe = exe

        self.dm_faiss = DataModule(
            ctx=self.ctx,
            state_key="faiss_store",
        )
        self.dm_rag = DataModule(
            ctx=self.ctx,
            state_key="rag_pipeline",
        )

        self.local_llm = LLMGenerator(
            ctx=self.ctx,
        )

    def run(self):
        steps = [
            RAGRetrievalBuilder(
                ctx=self.ctx,
                dm=self.dm_faiss,
                llm=self.local_llm,
            ),
            RAGResponseGenerator(
                ctx=self.ctx,
                dm=self.dm_rag,
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
