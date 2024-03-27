from importlib import import_module as __import_module
from common.cache import CacheType
from singleton_pattern import load_config as __load_config
from data_generator.Base import BaseDataGenerator
from common.context import Context

def create_tensor_data_generator(dataset_type):
    __config = __load_config.get_config()

    method_name = __config['method']
    model_file = __import_module(f'data_generator.{method_name}')
    DataGenerator = getattr(model_file,'DataGenerator')
    return DataGenerator(dataset_type)

def get_train_data_generator() -> BaseDataGenerator:
    if Context.train_generator == None:
        Context.train_generator = create_tensor_data_generator(CacheType.TRAIN)
    return Context.train_generator
def get_test_data_generator() -> BaseDataGenerator:
    if Context.test_generator == None:
        Context.test_generator = create_tensor_data_generator(CacheType.TEST)
    return Context.test_generator
