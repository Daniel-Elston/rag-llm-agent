from __future__ import annotations


from config.pipeline_context import PipelineContext

from config.settings import Params

from transformers import AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class LLMPipeline:
    def __init__(
        self, ctx: PipelineContext,
    ):
        self.ctx = ctx
        self.params: Params = ctx.settings.params
        
    def hf_gen_pipeline(self):
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