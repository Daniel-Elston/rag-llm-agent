from __future__ import annotations

import logging

from utils.file_access import FileAccess
from config.state_init import StateManager
from src.data.data_module import DataModule


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
        self, state: StateManager,
        dm: DataModule,
    ):
        self.state = state
        self.dm = dm
        self.config = self.state.model_config

    def __call__(self):
        self._execute_test_query()

    def _execute_test_query(self):
        qa_chain = self.state.data_state.get("qa_chain")
        test_query = self._get_test_query()
        response = self._generate_response(qa_chain, test_query)
        # self._log_generated_response(test_query, response)
        
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
        logging.debug(log_entry)
        FileAccess.save_file(
            log_entry, self.state.paths.get_path("generated-answers")
        )
