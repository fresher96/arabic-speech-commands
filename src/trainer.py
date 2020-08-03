import os
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
import numpy as np
import random


class ModelTrainer():

    def __init__(self, model, dataloader, args):
        self.model = model
        self.args = args
        self.data = dataloader

        self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum);
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)

    def train_one_epoch(self, epoch):

        self.model.train()
        train_loader = self.data['train']

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), leave=True, total=len(train_loader)):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            if (batch_idx + 1) % self.args.frq_log == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item()))

    def train(self):

        best = 0

        print(">> Training %s" % self.model.name)
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)
            res = self.test()
            if res['loss'] > best:
                best = res['loss']
                # self.save_weights(self.epoch)
        print(">> Training model %s.[Done]" % self.model.name)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            test_loader = self.data['val']
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))

        res = {'loss': test_loss, 'acc': accuracy}

        return res


