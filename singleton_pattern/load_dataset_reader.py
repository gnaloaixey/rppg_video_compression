from importlib import import_module as __import_module
from singleton_pattern import load_config as __load_config
from dataset_reader.Base import BaseDatasetReader
from common.context import Context
def create_train_reader(dataset_type) -> BaseDatasetReader:
    config = __load_config.get_config()
    loader_name = config[dataset_type]['dataset']['loader']
    loader_file = __import_module(f'dataset_reader.{loader_name}')
    DataseReader = getattr(loader_file,'DatasetReader')
    return DataseReader(dataset_type)

def get_train_reader() -> BaseDatasetReader:
    if Context.train_reader == None:
        Context.train_reader = create_train_reader('train')
    return Context.train_reader
def get_test_reader() -> BaseDatasetReader:
    if Context.test_reader == None:
        Context.test_reader = create_train_reader('test')
    return Context.test_reader
