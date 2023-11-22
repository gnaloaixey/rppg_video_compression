import pickle
from os import path,listdir,makedirs
import shutil
import yaml
import torch
from torch.utils.data import Dataset
from util.read_file import generate_dict_hash, generate_file_hash
from singleton_pattern.load_config import get_config
import enum

class CacheType(enum.Enum):
    TEST = 1
    TRAIN = 2
    MODEL = 3
    RUNTIME = 4
    RESULT = 5

cache_root = 'cache'
class Cache:
    @staticmethod
    def clear_useless_cache():
        __config_root = 'config'
        __cache_root = 'cache'
        if not path.exists(__cache_root) or not path.exists(__config_root):
            return
        config_file_names = [name for name in listdir(__config_root) if path.isfile(path.join(__config_root, name))]
        config_file_hashs = [generate_file_hash(path.join(__config_root,name)) for name in config_file_names]

        cache_file_names = [name for name in listdir(__cache_root) if path.isdir(path.join(__cache_root, name))]
        for name in cache_file_names:
            if name not in config_file_hashs:
                shutil.rmtree(path.join(__cache_root,name))   
                pass         
        pass
    file_path = None
    runtime_info_name = 'info.yaml'
    model_name = 'model.pkl'
    def __init__(self,cache_type:CacheType) -> None:
        # Cache.clear_useless_cache()
        hash_name = self.__get_hash_name(cache_type)
        self.file_path = path.join(cache_root,cache_type.name,hash_name)
        makedirs(self.file_path, exist_ok=True)
        print(f'cache path:{self.file_path}')
    def __get_hash_name(self,cache_type:CacheType):
        content = get_config()
        if cache_type == CacheType.MODEL:
            return ''
        if cache_type == CacheType.TEST or cache_type == CacheType.TRAIN:
            dataset_type = cache_type.name.lower()
            compression_config = content['data_format'].get('compression',{})
            compress_config = compression_config if compression_config.get('enble',False) else {}
            return generate_dict_hash({
                'fps':content['data_format']['fps'],
                'slice_interval':content['data_format']['slice_interval'],
                'step':content['data_format']['step'],
                'loader':content[dataset_type]['dataset']['loader'],
                'path':content[dataset_type]['dataset']['path'],
                **compress_config,
            })
        if cache_type == CacheType.RUNTIME:
            return generate_dict_hash(content)
        raise Exception('cache type error')
    def exist(self) -> bool:
        return path.exists(self.file_path) and path.isdir(self.file_path)

    def free(self):
        shutil.rmtree(self.file_path)

    def save(self,X,y,index):
        try:
            dir = path.join(self.file_path,str(index))
            makedirs(dir, exist_ok=True)
            with open(path.join(dir,'X.pkl'), 'wb') as file:
                pickle.dump(X, file)
                file.close()
            with open(path.join(dir,'y.pkl'), 'wb') as file:
                pickle.dump(y, file)
                file.close()
        except Exception as e:
            print(f"Error saving DataLoader,index : {index} , {e}")
    def read(self,index):
        with open(path.join(self.file_path,str(index),'X.pkl'), 'rb') as file:
            X = pickle.load(file)
            file.close()
        with open(path.join(self.file_path,str(index),'y.pkl'), 'rb') as file:
            y = pickle.load(file)
            file.close()
        return X,y
    def size(self) -> int:
        if not self.exist():
            return 0
        subdirectories = [d for d in listdir(self.file_path) if path.isdir(path.join(self.file_path, d))]
        return len(subdirectories)
    def save_cache_info(self,key,value):
        cache_data = {key: value}
        with open(path.join(self.file_path,self.runtime_info_name), 'w',encoding='utf-8') as file:
            yaml.dump(cache_data, file, default_flow_style=False)
            file.close()
    def save_model(self,model:torch.nn.Module):
        with open(path.join(self.file_path,self.model_name), 'wb') as file:
            pickle.dump(model, file)
            file.close()
    def read_model(self)->torch.nn.Module:
        model_path = path.join(self.file_path,self.model_name)
        if not path.exists(model_path):
            return None
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            file.close()
        return model
    def get_cache_info(self,key):
        with open(path.join(self.file_path,self.runtime_info_name), 'r',encoding='utf-8') as file:
            cache_data = yaml.safe_load(file)
            file.close()
        if cache_data and key in cache_data:
            return cache_data[key]
        else:
            return None
class CacheDataset(Dataset):
    __cache:Cache
    __tensor_X = None
    __tensor_y = None
    def __init__(self, cache:Cache,load_to_memory=False):
        self.__cache = cache
        if load_to_memory:
            tensor_X = list()
            tensor_y = list()
            print('loading to memory...')
            for index in range(cache.size()):
                X,y = self.__cache.read(index)
                tensor_X.append(X)
                tensor_y.append(y)
            self.__tensor_X = tensor_X
            self.__tensor_y = tensor_y
            print('load end')
    def __len__(self):
        if self.is_load_to_memory():
            return len(self.__tensor_y)
        return self.__cache.size()
    def __getitem__(self, index):
        if self.is_load_to_memory():
            return self.__tensor_X[index],self.__tensor_y[index]
        X,y = self.__cache.read(index)
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        return X,y
    def is_load_to_memory(self):
        return self.__tensor_y is not None and self.__tensor_X is not None
