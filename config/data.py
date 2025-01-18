from __future__ import annotations

import attr
import logging
from pprint import pformat

# def default_labels():
#     return ["cat", "dog", "go", "happy", "left", "no", "off", "on", "yes", "zero"]


@attr.s
class DataConfig:
    chunk_size: int = attr.ib(default=1200)
    chunk_overlap: int = attr.ib(default=100)
    separators: list = attr.ib(default=["\n\n", "\n", ".", ";", ",", " ", ""])
    # overwrite: bool = attr.ib(default=True)
    # save_fig: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        attr_dict = attr.asdict(self)
        logging.debug(f"DataConfig:\n{pformat(attr_dict)}\n")

class DataState:
    def __init__(self):
        self._state = {}

    def set(self, key, value):
        """Store a value in the state"""
        logging.debug(f"Setting {key} to memory")
        self._state[key] = value

    def get(self, key):
        """Retrieve a value from the state"""
        return self._state.get(key)

    def clear(self):
        """Clear all state"""
        self._state = {}
