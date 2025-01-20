from __future__ import annotations

from config.pipeline_context import PipelineContext
from utils.execution import TaskExecutor
from src.data.data_module import DataModule

from config.settings import Config, Params

from src.data.embed import BuildVectorStore

from langchain_huggingface import HuggingFaceEmbeddings


class VectorStorePipeline:
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
        
        self.dm_chunk_docs = DataModule(
            ctx=self.ctx,
            state_key="chunk_docs_all",
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.params.embedding_model_name
        )


    def build_vector_store(self):
        steps = [
            BuildVectorStore(
                ctx = self.ctx,
                dm = self.dm_chunk_docs,
                embeddings = self.embeddings
            )
        ]
        self.exe._execute_steps(steps, stage="parent")
