

from util.static_var import StaticVar
# Open and read the configuration file

def get_config_hash():
    if StaticVar.config_hash == None:
        raise RuntimeError(f"Run init first.")
    return StaticVar.config_hash

def get_config()->dict:
    if StaticVar.config == None:
        raise RuntimeError(f"Run init_config first.")
    return StaticVar.config
def get_non_dnn_method_list()->list:
    if StaticVar.non_dnn_method_list == None:
        # Method type, determine if training is needed
        with open('non_dnn_method_list.txt', 'r') as file:
            StaticVar.non_dnn_method_list = [line.strip() for line in file if line.strip()]
    return StaticVar.non_dnn_method_list
