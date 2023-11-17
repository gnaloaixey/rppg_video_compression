from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config
from method.TrainableMethod import TrainableMethod
__map = dict()

def get_model() -> TrainableMethod:
    __config = __load_config.get_config()
    method_name = __config['method']
    if method_name not in __map:
        model_file = __import_module(f'method.{method_name}')
        Model = getattr(model_file,method_name)
        __map[method_name] = Model()
    return  __map.get(method_name)
