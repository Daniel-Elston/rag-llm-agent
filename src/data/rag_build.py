from __future__ import annotations

from config.pipeline_context import PipelineContext
from src.data.data_module import DataModule

from config.settings import Config, Params
from config.states import DataState

from transformers import AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.documents import Document

from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter


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
        self, ctx: PipelineContext,
        dm: DataModule,
    ):
        self.ctx = ctx
        self.faiss_store = dm.load()
        self.params: Params = ctx.settings.params
        self.config: Config = ctx.settings.config
        self.data_state: DataState = ctx.states.data


    def __call__(self):
        self.initialise()

    def initialise(self):
        """Initialize the RAG pipeline components and save to state."""
        retriever = self._build_retriever()
        local_llm = self._build_local_llm()
        rag_pipeline = self._build_rag_pipeline(retriever, local_llm)
        self._save_helper(rag_pipeline)
        
    def _build_retriever(self):
        """Build a retriever from the FAISS store"""
        return self.faiss_store.as_retriever()
    
    def _build_local_llm(self):
        """Build a local huggingface pipeline for generation"""
        model_name = self.params.language_model_name
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
    
    def _build_rag_pipeline(self, retriever, local_llm):
        result_docs = retriever.invoke("test query")
        for i in range(len(result_docs)):
            assert type(result_docs[i]) is Document
        
        retrieval_step = self.make_retrieval_step(retriever)
        docs_to_str_step = self.make_docs_to_str_step()
        prompt_step = self.make_prompt_step()
        output_finaliser_step = self.make_output_finaliser_step()
        
        rag_pipeline = (
            retrieval_step
            | RunnableMap({
                "docs": itemgetter("docs"),
                "question": itemgetter("question"),
            })
            | RunnableMap({
                "context": (itemgetter("docs") | docs_to_str_step),
                "question": itemgetter("question"),
                "source_docs": itemgetter("docs")
            })
            | prompt_step
            | local_llm
            | output_finaliser_step
        )
        return rag_pipeline
    
    @staticmethod
    def make_retrieval_step(retriever):
        """
        INPUT: {"query": ...}
        OUTPUT: {"docs", "question"}.
        """
        return RunnableLambda(
            lambda inputs: {
                "docs": retriever.invoke(inputs["query"]),
                "question": inputs["query"],
            },
            name="retrieval",
        )

    @staticmethod
    def make_docs_to_str_step():
        """Convert the list of docs to a single string."""
        return RunnableLambda(
            lambda docs: "\n\n".join(doc.page_content for doc in docs),
            name="docs_to_str"
        )

    @staticmethod
    def make_prompt_step():
        """
        Return a ChatPromptTemplate (or any prompt). 
        Prompt expects {"context", "question"} as input.
        """
        return ChatPromptTemplate.from_template(
            "Context:\n{context}\n\nUser Question: {question}\n\n"
            # "Please provide a concise answer referencing the context above."
            # "If something isn't in the context, say you do not know."
            "Output some key words from the context."
        )

    @staticmethod
    def make_output_finaliser_step():
        """
        Merges the LLM output with "source_docs"
        OUTPUT: Final dictionary:
            {"result": <text>, "source_documents": <list of docs>}
        """
        return RunnableLambda(
            lambda inputs: {
                "result": inputs,           
                # "source_documents": inputs["source_docs"] 
            },
            name="output_finaliser"
        )
    
    def _save_helper(self, rag_pipeline):
        self.data_state.set("rag_pipeline", rag_pipeline)