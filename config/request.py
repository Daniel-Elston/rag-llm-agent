from __future__ import annotations

import attr
import logging
from pprint import pformat

def default_labels():
    return ["cat", "dog", "go", "happy", "left", "no", "off", "on", "yes", "zero"]


@attr.s
class RequestConfig:
    pass
    
    def __attrs_post_init__(self):
        # attr_dict = attr.asdict(self)
        # logging.debug(f"RequestConfig:\n{pformat(attr_dict)}\n")
        pass


class RequestState:
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