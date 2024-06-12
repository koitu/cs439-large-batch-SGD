import os
from datetime import datetime

CIFAR100_STATS = ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])

# TODO: WHAT IN THE WORLD IS GOING ON WITH THE TRAINING???
# TODO: WHY IS THE RESULT SO DIFFERENT FROM THE RESULTS IN THE FIRST LIBRARY

# CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
# CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

# directory to save weights file
CHECKPOINT_DIR = 'checkpoint'

# total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

# TODO: what kind of training rates do we want to try???
# warmup training
# different version of the step rate
# linear scaling learning rate
# cosine learning rate decay

# bonus: try this stuff with nesterov mometum (which is somehow different from ragular momemtum...)

# initial learning rate
# INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# tensorboard log dir
LOG_DIR = 'logs'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
