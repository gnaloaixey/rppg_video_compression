from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config
from dataset_reader.Base import DatasetReader
__train_reader = None
__test_reader = None
def __generate_train_reader(type) -> DatasetReader:
    __config = __load_config.get_config()
    loader_name = __config[type]['dataset']['loader']
    loader_file = __import_module(f'dataset_reader.{loader_name}')
    DataseReader = getattr(loader_file,'DatasetReader')
    return DataseReader(__config[type]['dataset']['path'])

def get_train_reader() -> DatasetReader:
    global __train_reader
    if __train_reader == None:
        __train_reader = __generate_train_reader('train')
    return __train_reader
def get_test_reader() -> DatasetReader:
    global __test_reader
    if __test_reader == None:
        __test_reader = __generate_train_reader('test')
    return __test_reader
