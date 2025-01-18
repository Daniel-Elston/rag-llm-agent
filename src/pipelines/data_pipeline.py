from __future__ import annotations

from config.state_init import StateManager
from utils.execution import TaskExecutor
from src.data.data_dict import NoDataDict
from src.data.data_module import DataModule
from src.data.process import Processor

class DataPipeline:
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe
        self.config = state.data_config
        self.dd = NoDataDict()

        self.data = DataModule(
            data_path=self.state.paths.get_path("raw"),
            data_dict=self.dd,
        ).load()


    def __call__(self):
        steps = [
            Processor(self.config, self.data).run,
        ]
        self.exe._execute_steps(steps, stage="parent")
