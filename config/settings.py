from __future__ import annotations

import attr
import logging
from pprint import pformat


@attr.s
class Config:
    write_output: bool = attr.ib(default=True)
    overwrite: bool = attr.ib(default=True)
    save_fig: bool = attr.ib(default=True)
    
    def __attrs_post_init__(self):
        attr_dict = attr.asdict(self)
        logging.debug(f"DataConfig:\n{pformat(attr_dict)}\n")

@attr.s
class Params:
    chunk_size: int = attr.ib(default=1000)    
    chunk_overlap: int = attr.ib(default=50)
    truncation: bool = attr.ib(default=True)
    max_input_seq_length: int = attr.ib(default=512)
    max_output_seq_length: int = attr.ib(default=512)
    separators: list = attr.ib(
        default=["\n\n", "\n", ".", ";", ",", " ", ""]
    )
    embedding_model_name: str = attr.ib(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    language_model_name: str = attr.ib(
        default="google/flan-t5-base"
    )

    def __attrs_post_init__(self):
        attr_dict = attr.asdict(self)
        logging.debug(f"ModelConfig:\n{pformat(attr_dict)}\n")

@attr.s
class HyperParams:
    pass

    def __attrs_post_init__(self):
        # attr_dict = attr.asdict(self)
        # logging.debug(f"ModelConfig:\n{pformat(attr_dict)}\n")
        pass
    

@attr.s
class Settings:
    """config, params, hyper_params"""
    config: Config = attr.ib(factory=Config)
    params: Params = attr.ib(factory=Params)
    hyper_params: HyperParams = attr.ib(factory=HyperParams)
    
    def __attrs_post_init__(self):
        attr_dict = attr.asdict(self)
        logging.debug(f"ExperimentConfig:\n{pformat(attr_dict)}\n")