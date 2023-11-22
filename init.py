from util.static_var import StaticVar
from util.read_file import generate_file_hash,read_yaml_file
import yaml as __yaml
import os



__root_dir = 'config'
__default_config_name = 'config.yaml'
def run(config_name = __default_config_name):
    StaticVar.clear_var()
    global __root_dir
    global __default_config_name
    if config_name == None:
        config_name = __default_config_name
    config_path = os.path.join(__root_dir, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"file '{config_path}' dose not exist.")
    StaticVar.config = read_yaml_file(config_path)
    StaticVar.config_hash = generate_file_hash(config_path)