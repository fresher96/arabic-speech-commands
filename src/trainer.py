import os;
from comet_ml import Experiment
import torch;
from torch import nn;
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
import numpy as np;
import random;


class ModelTrainer():

    def __init__(self, model, dataloader, args):
        self.model = model
        self.args = args
        self.data = dataloader
        self.metric = args.metric;

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device);

        if(args.optimizer == 'sgd'):
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum);
        elif(args.optimizer == 'adam'):
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999));
        else:
            raise Exception('--optimizer should be one of {sgd, adam}');

        if(not args.test):
            self.experiment = Experiment(api_key=args.comet_key,
                                         project_name=args.comet_project, workspace=args.comet_workspace)

            self.experiment.log_parameters(vars(args));
            self.experiment.set_model_graph(str(self.model));
            self.experiment.set_name(args.name);

    def train_one_epoch(self, epoch):

        self.model.train()
        train_loader = self.data['train'];
        train_loss = 0
        correct = 0

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), leave=True, total=len(train_loader)):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item()
            train_loss += loss.item();
            correct += acc;

            # loss = loss.item() / len(data);
            acc = 100. * acc / len(data);

            self.experiment.log_metric('batch_loss', loss, batch_idx);
            self.experiment.log_metric('batch_acc', acc, batch_idx);

            if (batch_idx + 1) % self.args.frq_log == 0:
                self.experiment.log_metric('log_loss', loss, batch_idx);
                self.experiment.log_metric('log_acc', acc, batch_idx);
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                    epoch + 1, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss, acc))

        train_loss /= len(train_loader.dataset)
        acc = 100. * correct / len(train_loader.dataset);

        epoch_log_step = (epoch + 1) * (len(train_loader) - 1);
        self.experiment.log_metric('epoch_loss', train_loss, epoch_log_step);
        self.experiment.log_metric('epoch_acc', acc, epoch_log_step);

        print('Epoch: {} [Done]\tLoss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)'.format(
            epoch + 1, train_loss, correct, len(train_loader.dataset), acc))

    def train(self):

        best = 0

        print(">> Training %s" % self.model.name)
        for epoch in range(self.args.nepoch):
            with self.experiment.train():
                self.train_one_epoch(epoch)

            with self.experiment.test():
                print("\nvalidation...");
                res = self.val(self.data['val'])
                self.experiment.log_metrics(res, step=(epoch + 1) * (len(self.data['train']) - 1));

            if res[self.metric] > best:
                best = res[self.metric]
                # self.save_weights(self.epoch)

        print(">> Training model %s.[Done]" % self.model.name)

    def val(self, val_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, leave=True, total=len(val_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset);

        print('Evaluation: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(val_loader.dataset), accuracy))

        res = {'loss': test_loss, 'acc': accuracy};
        return res;

    def test(self):
        # load_wights();
        print('\ntesting....');
        res = self.val(self.data['test']);
