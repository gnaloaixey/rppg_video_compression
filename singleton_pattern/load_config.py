import yaml as __yaml
import os
from util.read_file import generate_file_hash,read_yaml_file
# Open and read the configuration file

__config  = None
__non_dnn_method_list = None

__root_dir = 'config'
__default_config_name = 'config.yaml'
__config_hash = None
def init_config(config_name = __default_config_name):
    global __config
    global __config_hash
    global __root_dir
    if not __config == None:
        raise RuntimeError(f"Can only run init_config once,please restart kernel.")
    if config_name == None:
        config_name = __default_config_name
    config_path = os.path.join(__root_dir, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"file '{config_path}' dose not exist.")
    __config = read_yaml_file(config_path)
    __config_hash = generate_file_hash(config_path)

def get_config_hash():
    if __config_hash == None:
        raise RuntimeError(f"Run init_config first.")
    return __config_hash

def get_config():
    if __config == None:
        raise RuntimeError(f"Run init_config first.")
    return __config
def get_non_dnn_method_list():
    global __non_dnn_method_list
    if __non_dnn_method_list == None:
        # Method type, determine if training is needed
        with open('non_dnn_method_list.txt', 'r') as __file:
            __non_dnn_method_list = [line.strip() for line in __file if line.strip()]
    return __non_dnn_method_list
