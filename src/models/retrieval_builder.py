from __future__ import annotations

from config.pipeline_context import PipelineContext
from src.data.data_module import DataModule

from config.settings import Config, Params
from config.states import DataState

from src.models.hf_llm import LLMGenerator

from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter


class RAGRetrievalBuilder:
    """
    Summary: 
        Builds the retrieval system and augments retrieved documents into RAG pipeline
        for later generation. Utilises FAISS for retrieval and a Hugging Face LLM for
        local text generation.\n
    Input: FAISS vector store ``data_state key: faiss_store``\n
    Output: RAG Pipeline ``data_state key: rag_pipeline``\n
    Steps:
        1) Load FAISS store from state\n
        2) Initialise the retriever from the FAISS store\n
        3) Build the RAG Pipeline chain by combining retrieval and augmentation steps\n
        4) Save RAG Pipeline chain to state
    """
    def __init__(
        self, ctx: PipelineContext,
        dm: DataModule,
        llm: LLMGenerator
    ):
        self.ctx = ctx
        self.faiss_store = dm.load()
        self.local_llm = llm.hf_gen_pipeline()
        self.params: Params = ctx.settings.params
        self.config: Config = ctx.settings.config
        self.data_state: DataState = ctx.states.data


    def __call__(self):
        self.initialise()

    def initialise(self):
        """Initialize the RAG pipeline components and save to state."""    
        retriever = self.faiss_store.as_retriever()
        rag_pipeline = self._build_rag_pipeline(retriever, self.local_llm)
        self._save_helper(rag_pipeline)

    def _build_rag_pipeline(self, retriever, local_llm):
        retrieval_step = self.make_retrieval_step(retriever)
        first_map_step = self._first_map_step()
        second_map_step = self._second_map_step()
        prompt_step = self.make_prompt_step()
        output_finaliser_step = self.make_output_finaliser_step()
        
        rag_pipeline = (
            retrieval_step
            | first_map_step
            | second_map_step
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
    def _first_map_step():
        """
        Maps the initial retrieval results to maintain "docs" and "question".
        INPUT: {"docs", "question"}
        OUTPUT: {"docs", "question"}
        """
        return RunnableMap({
            "docs": itemgetter("docs"),
            "question": itemgetter("question"),
        })

    @staticmethod
    def make_docs_to_str_step():
        """Convert the list of docs to a single string."""
        return RunnableLambda(
            lambda docs: "\n\n".join(doc.page_content for doc in docs),
            name="docs_to_str"
        )

    @staticmethod
    def _second_map_step():
        """
        Maps "docs" to "context" by converting docs to a string and maintains "source_docs".
        INPUT: {"docs", "question"}
        OUTPUT: {"context", "question", "source_docs"}
        """
        docs_to_str_step = RAGRetrievalBuilder.make_docs_to_str_step()
        return RunnableMap({
            "context": (itemgetter("docs") | docs_to_str_step),
            "question": itemgetter("question"),
            "source_docs": itemgetter("docs"),
        })

    @staticmethod
    def make_prompt_step():
        """
        Return a ChatPromptTemplate (or any prompt). 
        Prompt expects {"context", "question"} as input.
        """
        template = """
            Context:\n{context}\n\nUser Question: {question}\n\n
            Output some key words from the context.
        """
        return ChatPromptTemplate.from_template(template)

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
            },
            name="output_finaliser"
        )

    def _save_helper(self, rag_pipeline):
        self.data_state.set("rag_pipeline", rag_pipeline)