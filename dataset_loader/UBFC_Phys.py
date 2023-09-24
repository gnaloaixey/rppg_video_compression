import pandas as pd

class DatasetLoader:
    root = ''
    loader_name = 'UBFC_Phys'
    def __init__(self,root) -> None:
        self.root = root
        pass
    def load_data(self):
        return pd.DataFrame()
    def ppg_reader(self,path):
        df = pd.read_csv(path)
        return df