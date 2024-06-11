import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import config
from utils import get_model, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_weights, last_epoch, best_acc_weights


def train(epoch):

    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(model.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warmup:
            warmup_scheduler.step()

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        # load to device
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
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    # add information to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch',
                        required=True, type=str, help='model architecture')
    parser.add_argument('-n', '--name',
                        default=None, type=str, help='name of the model')
    parser.add_argument('-j', '--workers',
                        default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size',
                        default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate',
                        default=0.1, type=float, help='initial learning rate (default: 0.1)')
    parser.add_argument('--lg', '--learning-gamma',
                        default=0.2, type=float, help='learning rate decay (default: 0.2)')
    parser.add_argument('-e', '--epochs',
                        default=300, type=int, help='number of epochs to run (default: 300)')
    parser.add_argument('-w', '--warmup',
                        default=0, type=int, help='warm up before training phase (default: 0)')
    parser.add_argument('-m', '--momentum',
                        default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay',
                        default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--seed',
                        default=None, type=int, help='seed for initializing training')
    parser.add_argument('-r', '--resume',
                        action='store_true', default=False, help='resume training')
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
    cifar100_training_loader = get_training_dataloader(
        *config.CIFAR100_STATS,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        *config.CIFAR100_STATS,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # sgd with momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.MILESTONES, gamma=args.lg)

    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup)

    if args.name is None:
        checkpoint_path = str(os.path.join(config.CHECKPOINT_DIR, args.arch, config.TIME_NOW))
        writer = SummaryWriter(log_dir=str(os.path.join(config.LOG_DIR, args.arch, config.TIME_NOW)))

    else:
        checkpoint_path = str(os.path.join(config.CHECKPOINT_DIR, args.arch, args.name))
        writer = SummaryWriter(log_dir=str(os.path.join(config.LOG_DIR, args.arch, args.name)))

    # create a new tensorboard log file
    input_tensor = torch.Tensor(1, 3, 32, 32).to(device)
    writer.add_graph(model, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = str(os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth'))

    best_acc = 0.0

    # resume training from a checkpoint
    if args.resume:
        best_weights = best_acc_weights(checkpoint_path)

        if best_weights:
            weights_path = os.path.join(checkpoint_path, best_weights)
            print(f'found best acc weights file:{weights_path}')
            print('load best training file to test acc...')
            model.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(checkpoint_path)
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(checkpoint_path, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        model.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(checkpoint_path)

    # start training
    for epoch in range(1, config.EPOCH + 1):
        if epoch > args.warmup:
            scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # start to save the best performing model after learning rate decays to 0.01
        if epoch > config.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(arch=args.arch, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(model.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % config.SAVE_EPOCH:
            weights_path = checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(model.state_dict(), weights_path)

    writer.close()
