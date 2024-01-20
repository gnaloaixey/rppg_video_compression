from common.context import Context
# Open and read the configuration file

def get_config_hash():
    if Context.config_hash == None:
        raise RuntimeError(f"Run init first.")
    return Context.config_hash

def get_config()->dict:
    if Context.config == None:
        raise RuntimeError(f"Run common.init.run() first.")
    return Context.config
def get_non_dnn_method_list()->list:
    if Context.non_dnn_method_list == None:
        raise RuntimeError(f"Run common.init.run() first.")
    return Context.non_dnn_method_list
