import pickle
from os import path,listdir,makedirs
import shutil
import torch
from torch.utils.data import Dataset
from util.read_file import generate_file_hash

class Cache:
    @staticmethod
    def clear_useless_cache():
        __config_root = 'config'
        __cache_root = 'cache'
        if not path.exists(__cache_root) or not path.exists(__config_root):
            return
        config_file_name = [name for name in listdir(__config_root) if path.isdir(path.join(__config_root, name))]
        config_file_hash = [generate_file_hash(path.join(__config_root,name)) for name in config_file_name]

        cache_file_names = [name for name in listdir(__cache_root) if path.isdir(path.join(__cache_root, name))]
        for name in cache_file_names:
            if name not in config_file_hash:
                # shutil.rmtree(path.join(__cache_root,name))   
                pass         
        pass
    file_path = None
    def __init__(self,file_hash) -> None:
        __cache_root = 'cache'
        self.file_path = path.join(__cache_root,file_hash)
        
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
Cache.clear_useless_cache()
class CacheDataset(Dataset):
    __cache:Cache
    def __init__(self, cache:Cache):
        self.__cache = cache
    def __len__(self):
        return self.__cache.size()
    def __getitem__(self, index):
        X,y = self.__cache.read(index)
        X = torch.tensor(X, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)
        return X,y
