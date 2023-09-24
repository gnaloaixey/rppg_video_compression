from importlib import import_module as __import_module
from load import load_config as __load_config
__config = __load_config.get_config()

__map = dict()
def get_encoder(path):
    codec_name = __config['compression']['codec']
    if codec_name not in __map:
        encoder_file = __import_module(f'encoder.{codec_name}')
        CodecClass = getattr(encoder_file,codec_name)
        __map[codec_name] = CodecClass()
    return __map.get(codec_name)