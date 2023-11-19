import numpy as np
from singleton_pattern.load_config import get_config
class BaseDatasetReader:
    def __init__(self,dataset_type) -> None:
        config = get_config()
        self.root = config.get(dataset_type,None).get('dataset',None).get('path',None)
        self.loader_name = config.get(dataset_type,None).get('dataset',None).get('loader',None)
        pass
    def print_start_reading(self):
        print(f"Root Path:{self.root}\nStart Reading {self.loader_name} Dataset Directory...")
    def load_data() -> tuple:
        return (np.array(),np.array())
