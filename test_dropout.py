# -*- coding: utf-8 -*-
# @Time    : 2022/11/1 18:56
# @Author  : Chongming GAO
# @FileName: test_dropout.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import datasets, transforms

from tqdm import tqdm_notebook, tqdm


def build_dataset(dataset='MNIST', dataset_dir='./data', batch_size=100):
    dataset_ = {
        'MNIST': datasets.MNIST,
        'CIFAR10': datasets.CIFAR10
    }[dataset]

    transform = {
        'MNIST': transforms.ToTensor(),
        'CIFAR10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }[dataset]

    train_dataset = dataset_(root=dataset_dir,
                             train=True,
                             transform=transform,
                             download=True)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   num_workers=6,
                                   batch_size=batch_size,
                                   shuffle=True)

    test_dataset = dataset_(root=dataset_dir,
                            train=False,
                            transform=transform,
                            download=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  num_workers=6,
                                  batch_size=batch_size,
                                  shuffle=False)

    return train_loader, test_loader


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x

def dropout(p=None, dim=None, method='standard'):
    if method == 'standard':
        return nn.Dropout(p)
    elif method == 'gaussian':
        return GaussianDropout(p/(1-p))
    elif method == 'variational':
        return VariationalDropout(p/(1-p), dim)


class Net(nn.Module):
    def __init__(self,
                 image_dim=28 * 28,
                 dropout_method='standard'):
        super(Net, self).__init__()
        """3-Layer Fully-connected NN"""

        self.net = nn.Sequential(
            nn.Linear(image_dim, 500),
            dropout(0.2, 500, dropout_method),
            nn.ReLU(),
            nn.Linear(500, 100),
            dropout(0.5, 100, dropout_method),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def kl(self):
        kl = 0
        for name, module in self.net.named_modules():
            if isinstance(module, VariationalDropout):
                kl += module.kl().sum()
        return kl

    def forward(self, x):
        return self.net(x)


class Solver(object):
    def __init__(self, dropout_method='standard', dataset='MNIST', n_epochs=50, lr=0.005):
        self.n_epochs = n_epochs

        self.train_loader, self.test_loader = build_dataset(dataset, './data')

        self.image_dim = {'MNIST': 28 * 28, 'CIFAR10': 3 * 32 * 32}[dataset]

        self.dropout_method = dropout_method

        self.net = Net(
            image_dim=self.image_dim,
            dropout_method=dropout_method).cuda()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def train(self):
        self.net.train()

        for epoch_i in tqdm(range(self.n_epochs)):
            epoch_i += 1
            epoch_loss = 0
            epoch_kl = 0
            for images, labels in self.train_loader:
                images = Variable(images).view(-1, self.image_dim).cuda()
                labels = Variable(labels).cuda()

                logits = self.net(images)

                loss = self.loss_fn(logits, labels)

                if self.dropout_method == 'variational':
                    kl = self.net.kl()
                    total_loss = loss + kl / 10
                else:
                    total_loss = loss

                self.optimizer.zero_grad()
                total_loss.backward()

                self.optimizer.step()

                epoch_loss += float(loss.data)
                if self.dropout_method == 'variational':
                    epoch_kl += float(kl.data)

            if not self.dropout_method == 'variational':
                epoch_loss /= len(self.train_loader.dataset)
                print(f'Epoch {epoch_i} | loss: {epoch_loss:.4f}')

            else:
                epoch_loss /= len(self.train_loader.dataset)
                epoch_kl /= len(self.train_loader.dataset)
                print(f'Epoch {epoch_i} | loss: {epoch_loss:.4f}, kl: {epoch_kl:.4f}')

    def evaluate(self):
        total = 0
        correct = 0
        self.net.eval()
        for images, labels in self.test_loader:
            images = Variable(images).view(-1, self.image_dim).cuda()

            logits = self.net(images)

            _, predicted = torch.max(logits.data, 1)

            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        print(f'Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':

    # standard_solver = Solver('standard')
    # standard_solver.train()
    # standard_solver.evaluate()

    # gaussian_solver = Solver('gaussian')
    # gaussian_solver.train()
    # gaussian_solver.evaluate()

    variational_solver = Solver('variational')
    variational_solver.train()
    variational_solver.evaluate()
