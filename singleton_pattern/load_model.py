from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config
from method.Base import BaseMethod
from common.context import Context

def create_model() -> BaseMethod:
    __config = __load_config.get_config()
    method_name = __config['method']
    model_file = __import_module(f'method.{method_name}')
    Model = getattr(model_file,method_name)
    return Model()
def get_model() -> BaseMethod:
    __config = __load_config.get_config()
    method_name = __config['method']
    if method_name not in Context.model_map:
        Context.model_map[method_name] = create_model()
        return Context.model_map[method_name]
    return  Context.model_map.get(method_name)
