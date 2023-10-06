from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config
__config = __load_config.get_config()
__train_loader = None
__test_loader = None
def __generate_train_loader(type):
    loader_name = __config[type]['dataset']['loader']
    loader_file = __import_module(f'dataset_loader.{loader_name}')
    DatasetLoader = getattr(loader_file,'DatasetLoader')
    return DatasetLoader(__config[type]['dataset']['path'])

def get_train_loader():
    global __train_loader
    if __train_loader == None:
        __train_loader = __generate_train_loader('train')
    return __train_loader
def get_test_loader():
    global __test_loader
    if __test_loader == None:
        __test_loader = __generate_train_loader('test')
    return __test_loader