from __future__ import annotations

import logging

from config.state_init import StateManager
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.chunk_pipeline import ChunkPipeline
from src.pipelines.vector_pipeline import VectorStorePipeline
from utils.execution import TaskExecutor
from utils.project_setup import init_project


class MainPipeline:
    def __init__(
        self, state: StateManager,
        exe: TaskExecutor
    ):
        self.state = state
        self.exe = exe

    def run(self):
        """ETL pipeline main entry point."""
        steps = [
            DataPipeline(self.state, self.exe).prepare_data,
            ChunkPipeline(self.state, self.exe).chunk_data,
            VectorStorePipeline(self.state, self.exe).build_vector_store,
        ]
        self.exe._execute_steps(steps, stage="main")

if __name__ == "__main__":
    project_dir, project_config, state_manager, exe = init_project()
    try:
        logging.info(f"Beginning Top-Level Pipeline from ``main.py``...\n{"="*125}")
        MainPipeline(state_manager, exe).run()
    except Exception as e:
        logging.error(f"Pipeline terminated due to unexpected error: {e}", exc_info=True)
