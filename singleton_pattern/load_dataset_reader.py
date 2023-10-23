from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config

__train_loader = None
__test_loader = None
def __generate_train_reader(type):
    __config = __load_config.get_config()
    loader_name = __config[type]['dataset']['loader']
    loader_file = __import_module(f'dataset_reader.{loader_name}')
    DatasetLoader = getattr(loader_file,'DatasetReader')
    return DatasetLoader(__config[type]['dataset']['path'])

def get_train_reader():
    global __train_loader
    if __train_loader == None:
        __train_loader = __generate_train_reader('train')
    return __train_loader
def get_test_reader():
    global __test_loader
    if __test_loader == None:
        __test_loader = __generate_train_reader('test')
    return __test_loader
