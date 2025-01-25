from __future__ import annotations

from config.pipeline_context import PipelineContext
from src.data.data_module import DataModule
from src.models.chat_interface import RAGChatDashboard
from src.models.hf_llm import LLMGenerator
from src.models.memory_retrieval_builder import RAGConversationalBuilder
from utils.execution import TaskExecutor


class RAGConversationalPipeline:
    """
    Summary: Builds a conversational retrieval system with memory and provides
        a Gradio-based chat interface. Combines document
        retrieval, memory augmentation, and LLM-based generation.\n
        Input: FAISS vector store ``data_state key: faiss_store``\n
        Output: Gradio chat interface and conversational responses ``data_state key: convo_chain``
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
        self.dm_rag_conv = DataModule(
            ctx=self.ctx,
            state_key="convo_chain",
        )

        self.local_llm = LLMGenerator(
            ctx=self.ctx,
        )

    def run(self):
        steps = [
            RAGConversationalBuilder(
                ctx=self.ctx,
                dm=self.dm_faiss,
                llm=self.local_llm,
            ),
            RAGChatDashboard(
                ctx=self.ctx,
                dm=self.dm_rag_conv,
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
