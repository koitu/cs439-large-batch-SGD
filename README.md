# Experiments with Large Mini-batch SGD
To get plots for the runs we did, run the command below and visit [here](http://localhost:6006)
```
$ tensorboard --logdir='eval' --port=6006 --host='localhost'
```

## Dependencies
```
torch
torchvision
tensorboard
python 3.11
```

## Training
```
$ tensorboard --logdir='logs' --port=6006 --host='localhost'
$ python main.py --name s1_b0256 -s 1 -b 256
```
Most of the runs we did can be replicated by running `./run.sh`.
