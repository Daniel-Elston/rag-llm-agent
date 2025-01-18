from __future__ import annotations

import re
import unicodedata

from config.state_init import StateManager
from src.data.data_module import DataModule


class ProcessText:
    """
    Summary: Light NLP cleaning for document text
    Brief:
        - Standardise special characters if complex fonts
        - Remove excessive whitespace
        - Fix repeated punctuation (e.g. ". ." -> ".")
    """
    def __init__(
        self, state: StateManager,
        dm: DataModule,
    ):
        self.state = state
        self.dm = dm

    def __call__(self):
        documents = self.dm.load()
        for doc in documents:
            doc.page_content = self.clean_document_text(doc.page_content)
        self.state.data_state.set("proc_docs_all", documents)

    def clean_document_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\.\s*\.", ".", text)
        # Remove reference bracket [1], [2]...? Could be useful
        # text = re.sub(r"\[\d+\]", "", text)
        return text
