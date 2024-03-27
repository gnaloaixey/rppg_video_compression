from singleton_pattern.load_config import get_config_hash
from os import path
class BaseEncoder:
    def __init__(self) -> None:
        self.dir = path.join('cache',get_config_hash(),'encoder')
        pass