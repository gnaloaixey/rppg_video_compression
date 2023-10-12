import yaml as __yaml
import os
# Open and read the configuration file

__config  = None
__non_dnn_method_list = None

__root_dir = 'config'
__default_config_name = 'config.yaml'
def init_config(config_name):
    global __config
    global __root_dir
    if not __config == None:
        raise RuntimeError(f"Can only run init_config once,please restart kernel.")
    if config_name == None:
        config_name = __default_config_name
    config_path = os.path.join(__root_dir, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"file '{config_path}' dose not exist.")
    with open(config_path, 'r') as __yaml_file:
        __config = __yaml.load(__yaml_file, Loader=__yaml.FullLoader)



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
