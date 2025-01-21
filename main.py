from __future__ import annotations

import logging

from config.pipeline_context import PipelineContext
from utils.execution import TaskExecutor
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.chunk_pipeline import ChunkPipeline
from src.pipelines.vector_pipeline import VectorStorePipeline
from src.pipelines.retrieval_pipeline import RAGRetrievalPipeline
from src.pipelines.conversation_pipeline import RAGConversationalPipeline
from utils.project_setup import init_project


class MainPipeline:
    """RAG Pipeline main entry point."""
    def __init__(
        self, ctx: PipelineContext,
        exe: TaskExecutor
    ):
        self.ctx = ctx
        self.exe = exe

    def run(self):
        steps = [
            DataPipeline(self.ctx, self.exe).prepare_raw_data,
            ChunkPipeline(self.ctx, self.exe).chunk_documents_for_embedding,
            VectorStorePipeline(self.ctx, self.exe).build_vector_store,
            RAGRetrievalPipeline(self.ctx, self.exe).run,
            RAGConversationalPipeline(self.ctx, self.exe).run,
        ]
        self.exe._execute_steps(steps, stage="main")

if __name__ == "__main__":
    project_dir, project_config, ctx, exe = init_project()
    try:
        logging.info(f"Beginning Top-Level Pipeline from ``main.py``...\n{"="*125}")
        MainPipeline(ctx, exe).run()
    except Exception as e:
        logging.error(f"Pipeline terminated due to unexpected error: {e}", exc_info=True)
