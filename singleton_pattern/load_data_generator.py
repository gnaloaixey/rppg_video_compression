from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config
from data_generator.Base import BaseDataGenerator
from util.static_var import StaticVar

def create_tensor_data_generator(dataset_type):
    __config = __load_config.get_config()

    method_name = __config['method']
    model_file = __import_module(f'data_generator.{method_name}')
    DataGenerator = getattr(model_file,'DataGenerator')
    return DataGenerator(dataset_type)

def get_train_data_generator() -> BaseDataGenerator:
    if StaticVar.train_generator == None:
        StaticVar.train_generator = create_tensor_data_generator('train')
    return StaticVar.train_generator
def get_test_data_generator() -> BaseDataGenerator:
    if StaticVar.test_generator == None:
        StaticVar.test_generator = create_tensor_data_generator('test')
    return StaticVar.test_generator
