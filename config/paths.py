from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union
from pprint import pformat


paths_store = {
    "raw": Path("data/raw/heart.csv"),
    "sdo": Path("data/sdo/heart.parquet"),
    
    "load1_transform": Path("data/raw/heart1.csv"),
    "load2_fetch": Path("data/raw/heart2.csv"),
}


@dataclass
class PathsConfig:
    paths: Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        self.paths = {k: Path(v) for k, v in paths_store.items()}
        logging.debug(f"PathsConfig:\n{pformat(self.paths)}\n")

    def get_path(self, key: Optional[Union[str, Path]]) -> Optional[Path]:
        if key is None:
            return None
        if isinstance(key, Path):
            return key
        return self.paths.get(key)
