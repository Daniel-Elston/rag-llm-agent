# RAG-LLM Agent

### Summary:
Utilising LangChain and HuggingFace transformers, a Retrieval-Augmented-Generation system has been built to optimise generative responses of LLMs.

### Overview
- Loads both locally stored, and arxiv requested documents, performs light NLP processing
- Chunks documents and stores in-memory FAISS vector database
- Builds RAG pipeline and utilises LLM for genAI responses

### Current Drawbacks
- Prompt techniques are basic, due to basic pre-trained LLM models used, with small max sequence lengths
- Project built to easily switch to more sophisticated openAI model and improve functionality greatly.