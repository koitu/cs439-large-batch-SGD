from vgg import vgg_models
from resnet import resnet_models

# merge all models into a single models dict
models = (
    vgg_models |
    resnet_models
)
