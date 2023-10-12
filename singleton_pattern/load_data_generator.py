from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config

__generator = None
def get_tensor_data_generator():
    __config = __load_config.get_config()
    global __generator
    if __generator == None:
        method_name = __config['method']
        model_file = __import_module(f'data_generator.{method_name}')
        DataGenerator = getattr(model_file,'DataGenerator')
        __generator = DataGenerator()
    return __generator
