from __future__ import annotations

from config.state_init import StateManager
from config.model import ModelConfig
from src.data.data_module import DataModule

from transformers import AutoTokenizer

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class RAGBuilder:
    """
    Summary: 
        Builds the retrieval system and augments retrieved documents into QA chains
        for later generation. Utilises FAISS for retrieval and a Hugging Face LLM for
        local text generation.\n
    Input: FAISS store ``data_state key: faiss_store``\n
    Output: QA chain ``data_state key: qa_chain``\n
    Steps:
        1) Load FAISS store from state\n
        2) Build a retriever\n
        3) Build a HuggingFace pipeline (so we don't use OpenAI tokens)\n
        4) Build a RetrievalQA chain\n
        5) Test a single query and log/print result\n
    """
    def __init__(
        self, state: StateManager,
        model_config: ModelConfig,
        dm: DataModule,
    ):
        self.state = state
        self.dm = dm
        self.model_config = model_config
        self.faiss_store = self.dm.load()

    def __call__(self):
        self.initialise()

    def initialise(self):
        """Initialize the retrieval QA pipeline components."""
        retriever = self._build_retriever()
        local_llm = self._build_local_llm()
        qa_chain = self._build_retrieval_qa_chain(retriever, local_llm)
        # self.state.data_state.set("qa_chain", qa_chain)
        self._save_helper(qa_chain)

    def _build_retriever(self):
        """Build a retriever from the FAISS store"""
        return self.faiss_store.as_retriever()
        
    def _build_local_llm(self):
        """Build a local huggingface pipeline for generation"""
        model_name = self.model_config.language_model_name
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            truncation=True,
            model_max_length=512
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)

    def _build_retrieval_qa_chain(self, retriever, local_llm):
        """"Build a retrieval QA chain"""
        return RetrievalQA.from_chain_type(
            llm=local_llm,
            retriever=retriever,
            return_source_documents=True
        )
    
    def _save_helper(self, qa_chain):
        self.state.data_state.set("qa_chain", qa_chain)