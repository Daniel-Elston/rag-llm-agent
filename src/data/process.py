from __future__ import annotations

from config.data import DataConfig
from src.data.data_module import DataModule

class Processor:
    def __init__(
        self, config:DataConfig,
        dataset:DataModule
    ):
        self.config = config
        self.dataset = dataset
    
    def run(self):
        self.remove_nans()
    
    def remove_nans(self):
        # print(self.dataset)
        df = self.dataset
        df = df.dropna()
        # print(df)


