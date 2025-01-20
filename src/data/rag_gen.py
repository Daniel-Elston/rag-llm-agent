from __future__ import annotations

from config.pipeline_context import PipelineContext
from utils.file_access import FileAccess
from src.data.data_module import DataModule

from config.paths import Paths
from config.settings import Config, Params
from config.states import DataState


class RAGGenerator:
    """
    Summary:
        Executes queries on the QA chain and generates responses. 
        Uses the retrieval and augmentation system built by the RAGBuilder 
        to answer questions with source attribution.\n
    Input: FAISS store ``data_state key: qa_chain``\n
    Output: Generated answers and sources
    """
    def __init__(
        self, ctx: PipelineContext,
        dm: DataModule,
    ):
        self.ctx = ctx
        self.dm = dm
        self.params: Params = ctx.settings.params
        self.config: Config = ctx.settings.config
        self.data_state: DataState = ctx.states.data
        self.paths: Paths = ctx.paths


    def __call__(self):
        self._execute_test_query()

    def _execute_test_query(self):
        qa_chain = self.data_state.get("qa_chain")
        test_query = self._get_test_query()
        response = self._generate_response(qa_chain, test_query)
        self._save_helper(test_query, response)
        
    def _get_test_query(self):
        """Retrieve a test query."""
        test_query_store = [
            "Give me a brief summary of quantum encryption.",
            "Give me a brief summary of challenges in quantum computing."
        ]
        return test_query_store[1]
        
    def _generate_response(self, qa_chain, query: str):
        """Generate a response from the QA chain for given query."""
        response = qa_chain.invoke({"query": query})
        return {
            "query": query,
            "answer": response["result"],
            "sources": [
                doc.metadata.get("source", "Unknown source")
                for doc in response["source_documents"]
            ]
        }
        
    def _log_generated_response(self, query, response):
        """Log the generated response to a file."""
        answer = response["answer"]
        sources = response["sources"]

        log_entry = (
            "=== Test Query ===\n"
            f"Q: {query}\n"
            f"A: {answer}\n"
            f"Sources used:\n"
            f"{sources}\n"
        )
        FileAccess.save_file(
            log_entry, 
            self.paths.get_path("generated-answers")
        )
        
    def _save_helper(self, query, response):
        if self.config.write_output:
            self._log_generated_response(query, response)