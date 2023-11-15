from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config
from data_generator.Base import BaseDataGenerator
__train_generator = None
__test_generator = None
def __get_tensor_data_generator(dataset_type):
    __config = __load_config.get_config()

    method_name = __config['method']
    model_file = __import_module(f'data_generator.{method_name}')
    DataGenerator = getattr(model_file,'DataGenerator')
    return DataGenerator(dataset_type)

def get_train_data_generator() -> BaseDataGenerator:
    global __train_generator
    if __train_generator == None:
        __train_generator = __get_tensor_data_generator('train')
    return __train_generator
def get_test_data_generator() -> BaseDataGenerator:
    global __test_generator
    if __test_generator == None:
        __test_generator = __get_tensor_data_generator('test')
    return __test_generator
