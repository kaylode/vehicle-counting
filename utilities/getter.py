from networks import *
from augmentations import *
from configs import *

from .cuda import get_devices_info

CACHE_DIR='./.cache'

def get_instance(config, **kwargs):
    # Inherited from https://github.com/vltanh/pytorch-template
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)
