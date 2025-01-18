from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from pprint import pformat
from utils.file_access import FileAccess
from config.state_init import StateManager
from typing import List, Dict, Callable, Optional, Union, Any

class DataModule:
    """
    Args:
        state (StateManager): The state manager to access in-memory data.
        state_key (str): Key to retrieve data from in-memory state.
        data_path (Path): Path to load data from a file.
        data_dict (Any): Data dictionary with transformations.
    """
    def __init__(
        self, state: StateManager, 
        state_key: str = None,
        data_path: Path = None,
        data_dict: Any = None,
    ):
        
        if not state_key and not data_path:
            raise ValueError("Either `state_key` or `data_path` must be provided.")
        
        self.state = state
        self.state_key = state_key
        self.data_path = data_path
        self.dd = data_dict

    def load(self):
        """
        Load data either from the in-memory state or a local file.
        """
        if self.state_key:
            data = self.load_data_from_state()
        elif self.data_path and self.data_path.exists():
            data = self.load_data_from_path()
        else:
            raise ValueError(
                f"Unable to load data. `state_key`: {self.state_key}, `data_path`: {self.data_path}"
            )
        data = self.apply_data_dict(data)
        return data
    
    def load_data_from_path(self):
        """Load data to memory using your file access layer"""
        return FileAccess.load_file(self.data_path)
    
    def load_data_from_state(self):
        return self.state.data_state.get(self.state_key)

    def apply_data_dict(self, df):
        """
        Apply data dictionary transformations in the correct order.
        """
        if not self.dd:
            return df
        transforms = self.dd.transforms_store()
        for func_name, func in transforms.items():
            df = func(df)
        return df

    def to_parquet(self, df: pd.DataFrame, output_path: Path = None):
        """Write df to parquet (if needed)"""
        if not output_path:
            output_path = self.data_path.with_suffix('.parquet')
        FileAccess.save_file(df, output_path)
        return df


def load_dataset(dm: DataModule) -> pd.DataFrame:
    if not hasattr(dm, "_loaded_data"):
        # try:
        dm._loaded_data = dm.load()
        # logging.debug(f"Loaded dataset from {dm.data_path} with shape {dm._loaded_data.shape}")
        logging.debug(f"Loaded dataset from {dm.data_path}")
        
        if dm._loaded_data is None:
            raise ValueError(f"Dataset at {dm.data_path} is empty.")
        # except FileNotFoundError as e:
        #     logging.error(f"File not found for {dm.data_path}: {e}")
        #     raise
    return dm._loaded_data