from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from pprint import pformat
from utils.file_access import FileAccess
from typing import List, Dict, Callable, Optional, Union, Any

class DataModule:
    def __init__(
        self, data_path: Path,
        data_dict: Any
    ):
        self.data_path = data_path
        self.dd = data_dict

    def load(self):
        df = self.load_data() 
        df = self.apply_data_dict(df)
        return df
    
    def load_data(self):
        """Load data to memory using your file access layer"""
        return FileAccess.load_file(self.data_path)

    def apply_data_dict(self, df):
        """Apply data dictionary transforms in the correct order"""
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
        try:
            dm._loaded_data = dm.load()
            logging.debug(f"Loaded dataset from {dm.data_path} with shape {dm._loaded_data.shape}")
            
            if dm._loaded_data.empty:
                raise ValueError(f"Dataset at {dm.data_path} is empty.")
        except FileNotFoundError as e:
            logging.error(f"File not found for {dm.data_path}: {e}")
            raise
    return dm._loaded_data