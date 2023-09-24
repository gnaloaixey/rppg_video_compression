import yaml as __yaml
# Open and read the configuration file
__config_name = 'config/config.yaml'
with open(__config_name, 'r') as __yaml_file:
    __config = __yaml.load(__yaml_file, Loader=__yaml.FullLoader)
# Method type, determine if training is needed
with open('non_dnn_method_list.txt', 'r') as __file:
    __non_dnn_method_list = [line.strip() for line in __file if line.strip()]

def get_config():
    return __config
def get_non_dnn_method_list():
    return __non_dnn_method_list