import sys

from .vgg import vgg_models
from .resnet import resnet_models

# merge all models into a single models dict
models = (
    vgg_models |
    resnet_models
)


def get_model(arch):
    """ return given network
    """
    try:
        return models[arch]()

    except KeyError:
        print('the network name you have entered is not supported')
        sys.exit()
