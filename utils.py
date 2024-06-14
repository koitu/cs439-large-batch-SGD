import numpy
import torchvision
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_training_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)

    cifar100_training_loader = DataLoader(
        cifar100_training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)

    cifar100_test_loader = DataLoader(
        cifar100_test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def compute_mean_std(cifar100_dataset):
    """
    compute the mean and std of cifar100 dataset

    :param cifar100_dataset: cifar100_training_dataset or cifar100_test_dataset from torch.utils.data
    :return: a tuple containing mean and std values over the dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        """warmup_training learning rate scheduler
        Args:
            optimizer: optimizer (e.g. SGD)
            total_iters: total_iters of warmup phase
        """
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
