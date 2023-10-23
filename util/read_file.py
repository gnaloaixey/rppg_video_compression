import hashlib as __hashlib
import yaml as __yaml

def read_yaml_file(file_path):
    with open(file_path, 'r',encoding="utf-8") as __yaml_file:
        info = __yaml.load(__yaml_file, Loader=__yaml.FullLoader)
        __yaml_file.close()
        return info
def generate_file_hash(file_path):
    with open(file_path, 'rb') as file:
        file_contents = file.read()
        file.close()
    return __hashlib.sha256(file_contents).hexdigest()