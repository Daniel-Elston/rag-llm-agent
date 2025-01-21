from __future__ import annotations

from config.pipeline_context import PipelineContext
from src.data.data_module import DataModule

from config.settings import Config, Params
from config.states import DataState

from src.models.hf_llm import LLMGenerator

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class RAGConversationalBuilder:
    """
    Summary: 
        Builds a conversational retrieval system with memory
        and augments retrieved documents into a conversational chain for later generation.
        Utilises FAISS for retrieval and a HuggingFace LLM for
        local text generation.\n
    Input: FAISS vector store ``data_state key: faiss_store``\n
    Output: RAG Pipeline ``data_state key: convo_chain``\n
    Steps:
        1) Load FAISS store from state\n
        2) Initialise the retriever from the FAISS store\n
        3) Build the conversational chain by integrating retrieval, memory, and augmentation\n
        4) Save the constructed conversational chain to state
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
        """Initialize the conversational chain + memory, save to data state."""
        retriever = self.faiss_store.as_retriever()
        convo_chain = self.build_conversational_chain(retriever, self.local_llm)
        self._save_helper(convo_chain)

    @staticmethod
    def build_conversational_chain(retriever, local_llm):
        """
        Creates a ConversationalRetrievalChain that:
            - uses a conversation buffer memory
            - fetches relevant doc chunks from 'retriever'
            - calls 'local_llm' to generate answers
            - returns source documents
        """
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        convo_chain = ConversationalRetrievalChain.from_llm(
            llm=local_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        return convo_chain
    
    def _save_helper(self, chain):
        self.data_state.set("convo_chain", chain)