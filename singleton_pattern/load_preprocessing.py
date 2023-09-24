from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config
__config = __load_config.get_config()

def get_tensor_data_generator():
    method_name = __config['method']
    model_file = __import_module(f'preprocessing.{method_name}')
    generate_tensor_data = getattr(model_file,'generate_tensor_data')
    return generate_tensor_data