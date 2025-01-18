from __future__ import annotations

import logging
from pprint import pformat

import attr


@attr.s
class ModelConfig:
    embedding_model_name: str = attr.ib(
        default="sentence-transformers/all-MiniLM-L6-v2")

    def __attrs_post_init__(self):
        attr_dict = attr.asdict(self)
        logging.debug(f"ModelConfig:\n{pformat(attr_dict)}\n")


class ModelState:
    def __init__(self):
        self._state = {}

    def set(self, key, value):
        """Store a value in the state"""
        self._state[key] = value

    def get(self, key):
        """Retrieve a value from the state"""
        return self._state.get(key)

    def clear(self):
        """Clear all state"""
        self._state = {}
