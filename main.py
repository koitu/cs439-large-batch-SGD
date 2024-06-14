import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import get_model
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR


def train(epoch):
    start = time.time()
    model.train()

    train_loss = 0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        # load some training images
        images = images.to(device)
        labels = labels.to(device)

        # train the model
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(
            f"Training epoch: {epoch} "
            f"[{batch_index * args.batch_size + len(images)}/{len(cifar100_training_loader.dataset)}]"
            f"\tLoss: {loss.item():0.4f}\tLR: {optimizer.param_groups[0]['lr']:0.6f}"
        )

        # update training loss for each iteration
        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
        writer.add_scalar('Train/Loss per Iteration', loss.item(), n_iter)
        train_loss += loss.item()

        if epoch <= args.warmup:
            warmup_scheduler.step()

    finish = time.time()

    print(f"Completed training epoch {epoch}, training time: {(finish - start):.2f}s")

    # update training loss for each epoch
    writer.add_scalar('Train/Average Loss per Epoch', train_loss / len(cifar100_training_loader), epoch)


@torch.no_grad()
def eval_training(epoch):

    start = time.time()
    model.eval()

    test_loss = 0.0  # loss function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        # load some testing images
        images = images.to(device)
        labels = labels.to(device)

        # evaluate the model
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if use_gpu:
        print(torch.cuda.memory_summary())

    print(
        f"Testing epoch: {epoch}, "
        f"Average loss: {test_loss / len(cifar100_test_loader.dataset):.4f}, "
        f"Accuracy: {correct.float() / len(cifar100_test_loader.dataset):.4f}, "
        f"Time consumed:{finish - start:.2f}s\n"
    )

    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Parameters/Learning Rate', optimizer.param_groups[0]['lr'], epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch',
                        default='resnet18', type=str, help='model architecture')

    parser.add_argument('-n', '--name',
                        required=True, type=str, help='name of the model')

    parser.add_argument('-s', '--schedule',
                        default=1, type=int, help='learning rate decay schedule (default: 1)')

    parser.add_argument('-b', '--batch-size',
                        default=128, type=int, help='mini-batch size (default: 128)')

    parser.add_argument('-w', '--warmup',
                        default=5, type=int, help='warm up before training phase (default: 5)')

    parser.add_argument('-m', '--momentum',
                        default=0.9, type=float, help='momentum (default: 0.9)')

    parser.add_argument('--wd', '--weight-decay',
                        default=5e-4, type=float, help='weight decay (default: 5e-4)')

    parser.add_argument('--nesterov',
                        action='store_true', help='Use nesterov momentum')

    parser.add_argument('-j', '--workers',
                        default=4, type=int, help='number of data loading workers (default: 4)')

    parser.add_argument('--seed',
                        default=None, type=int, help='seed for initializing training')

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # load the model
    model = get_model(args.arch)

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("WARNING: CUDA not available, using CPU instead")
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)

    # data preprocessing
    cifar100_stats = ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])

    cifar100_training_loader = get_training_dataloader(
        *cifar100_stats,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        *cifar100_stats,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # sgd with momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1 * (args.batch_size / 256),  # linear scaling rule
        nesterov=args.nesterov,
        momentum=args.momentum,
        weight_decay=args.wd)

    match args.schedule:
        case 1:  # decay by 1/2.5
            MAX_EPOCH = 150
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60, 90, 110, 130], gamma=0.4)

        case 2:  # decay by 1/5
            MAX_EPOCH = 200
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[60, 120, 160], gamma=0.2)

        case 3:  # decay by 1/10
            MAX_EPOCH = 300
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[150, 225], gamma=0.1)

        case 4:  # linear decay
            MAX_EPOCH = 200
            scheduler = optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=MAX_EPOCH, power=1.0)

        case 5:  # alternate linear decay
            MAX_EPOCH = 100
            scheduler = optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=MAX_EPOCH, power=1.0)

        case _:
            print("Invalid scheduler")
            sys.exit(1)

    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup)

    # create a new tensorboard log file
    writer = SummaryWriter(log_dir=str(os.path.join('logs', args.arch, args.name)))

    # start training
    for epoch in range(1, MAX_EPOCH + 1):
        if epoch > args.warmup:
            scheduler.step(epoch)

        train(epoch)
        eval_training(epoch)

    writer.close()
