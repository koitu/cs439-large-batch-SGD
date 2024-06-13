# Experiments with Large Mini-batch SGD
To get plots for the runs we did run the command below and visit [here](http://localhost:6006)
```
$ tensorboard --logdir='eval' --port=6006 --host='localhost'
```

```
torch
torchvision
tensorboard
python 3.11
```

## Usage
```
$ tensorboard --logdir='logs' --port=6006 --host='localhost'
$ python main.py --arch vgg16
```

for warmup training set `--warm` to 1 or 1, to prevent network diverging during early training phase.

The supported arch args are:
```
vgg11
vgg13
vgg16
vgg19
resnet18
resnet34
resnet50
resnet101
resnet152
```

## Training Details
Follows the hyperparameter settings in paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2)
- lr = 0.1 divide by 5 at 60th, 120th, 160th epochs
- train for 200 epochs with batchsize 128

Uses weight decay 5e-4 and Nesterov momentum of 0.9

Can also use the hyperparameters from paper [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548v1)
and [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896v2),
- lr = 0.1 and divide by 10 at 150th and 225th epochs
- training for 300 epochs with mini-batch size 128

You could decrease the mini-batch size to 64 or whatever if you dont have enough gpu memory.


## Results
The results got from a certain models, since the same hyperparameters are used to train all models
we could probably adjust them to get better results.

|dataset|network|params|top1 err|top5 err|epoch(lr = 0.1)|epoch(lr = 0.02)|epoch(lr = 0.004)|epoch(lr = 0.0008)|total epoch|
|:-----:|:-----:|:----:|:------:|:------:|:-------------:|:--------------:|:---------------:|:----------------:|:---------:|
|cifar100|vgg11_bn|28.5M|31.36|11.85|60|60|40|40|200|
|cifar100|vgg13_bn|28.7M|28.00|9.71|60|60|40|40|200|
|cifar100|vgg16_bn|34.0M|27.07|8.84|60|60|40|40|200|
|cifar100|vgg19_bn|39.0M|27.77|8.84|60|60|40|40|200|
|cifar100|resnet18|11.2M|24.39|6.95|60|60|40|40|200|
|cifar100|resnet34|21.3M|23.24|6.63|60|60|40|40|200|
|cifar100|resnet50|23.7M|22.61|6.04|60|60|40|40|200|
|cifar100|resnet101|42.7M|22.22|5.61|60|60|40|40|200|
|cifar100|resnet152|58.3M|22.31|5.81|60|60|40|40|200|
