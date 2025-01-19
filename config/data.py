from __future__ import annotations

import attr
import logging
from pprint import pformat


@attr.s
class DataConfig:
    write_output: bool = attr.ib(default=False)
    overwrite: bool = attr.ib(default=True)
    save_fig: bool = attr.ib(default=True)
    chunk_size: int = attr.ib(default=1000)    
    chunk_overlap: int = attr.ib(default=50)
    separators: list = attr.ib(default=["\n\n", "\n", ".", ";", ",", " ", ""])

    def __attrs_post_init__(self):
        attr_dict = attr.asdict(self)
        logging.debug(f"DataConfig:\n{pformat(attr_dict)}\n")

class DataState:
    def __init__(self):
        self._state = {}

    def set(self, key, value):
        """Store a value in the state"""
        logging.getLogger("file_access").file_track(
            f"SAVING ``{key}`` to {self.__class__.__name__} in memory")
        self._state[key] = value

    def get(self, key):
        """Retrieve a value from the state"""
        logging.getLogger("file_access").file_track(
            f"LOADING ``{key}`` from {self.__class__.__name__} in memory")
        return self._state.get(key)

    def clear(self):
        """Clear all state"""
        self._state = {}
