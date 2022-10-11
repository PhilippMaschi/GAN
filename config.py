from pathlib import Path
from abc import ABC
import os
# in this file we put certain path variables which are specific to our computer/system or project
# the other scripts will then use this file


class Config(ABC):
    def __init__(self):
        self.project_root: "Path" = self.setup_folder_path(Path(__file__).parent)
        # path where we store data in csv files
        self.path_2_data: "Path" = self.setup_folder_path(self.project_root / r"data/raw_data")
        # path to where outputs of the model should be stored
        self.output_path: "Path" = self.setup_folder_path(self.project_root / "data/output")
        # paths to where created figures should be stored
        self.fig_eda: "Path" = self.setup_folder_path(self.project_root / r"EDA/figures")
        self.fig_cluster: "Path" = self.setup_folder_path(self.project_root / r"clustering/figures")


    @staticmethod
    def setup_folder_path(folder_path) -> "Path":
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return Path(folder_path)