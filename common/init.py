from common.context import Context
from common.read_file import generate_file_hash,read_yaml_file
from common.cuda_info import print_info
import yaml as __yaml
import os


__root_dir = 'config'
__default_config_name = 'config.yaml'
def run(config_name = __default_config_name):
    Context.clear_var()
    global __root_dir
    global __default_config_name
    if config_name == None:
        config_name = __default_config_name
    config_path = os.path.join(__root_dir, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"file '{config_path}' dose not exist.")
    Context.config = read_yaml_file(config_path)
    with open('non_dnn_method_list.txt', 'r') as file:
        Context.non_dnn_method_list = [line.strip() for line in file if line.strip()]
    Context.config_hash = generate_file_hash(config_path)


    print_info()

    print(f'Method and DataGenerator Name: {Context.config["method"]}')