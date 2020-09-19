import os
from comet_ml import Experiment
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.ClassDict import ClassDict
from src.data import get_transform


class ModelTrainer:

    def __init__(self, model, dataloader, args):
        self.model = model
        self.args = args
        self.data = dataloader
        self.metric = args.metric

        if(dataloader is not None):
            self.frq_log = len(dataloader['train']) // args.frq_log

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)

        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                        betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
        else:
            raise Exception('--optimizer should be one of {sgd, adam}')

        if args.scheduler == 'set':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 10**(epoch/args.scheduler_factor))
        elif args.scheduler == 'auto':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                  factor=args.scheduler_factor, patience=5,
                                                                  verbose=True, threshold=0.0001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=0, eps=1e-08)

        self.experiment = Experiment(api_key=args.comet_key,
                                     project_name=args.comet_project, workspace=args.comet_workspace,
                                     auto_weight_logging=True, auto_metric_logging=False, auto_param_logging=False)

        self.experiment.set_name(args.name)
        self.experiment.log_parameters(vars(args))
        self.experiment.set_model_graph(str(self.model))


    def train_one_epoch(self, epoch):

        self.model.train()
        train_loader = self.data['train']
        train_loss = 0
        correct = 0

        comet_offset = epoch * len(train_loader)

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), leave=True, total=len(train_loader)):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item()
            train_loss += loss.item()
            correct += acc

            loss = loss.item() / len(data)
            acc = 100. * acc / len(data)

            comet_step = comet_offset + batch_idx
            self.experiment.log_metric('batch_loss', loss, comet_step, epoch)
            self.experiment.log_metric('batch_acc', acc, comet_step, epoch)

            if (batch_idx + 1) % self.frq_log == 0:
                self.experiment.log_metric('log_loss', loss, comet_step, epoch)
                self.experiment.log_metric('log_acc', acc, comet_step, epoch)
                print('Epoch: {} [{}/{}]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                    epoch + 1, (batch_idx + 1) * len(data), len(train_loader.dataset), loss, acc))

        train_loss /= len(train_loader.dataset)
        acc = 100. * correct / len(train_loader.dataset)

        comet_step = comet_offset + len(train_loader) - 1
        self.experiment.log_metric('loss', train_loss, comet_step, epoch)
        self.experiment.log_metric('acc', acc, comet_step, epoch)

        print('Epoch: {} [Done]\tLoss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)'.format(
            epoch + 1, train_loss, correct, len(train_loader.dataset), acc))

        return {'loss': train_loss, 'acc': acc}

    def train(self):

        self.log_cmd()
        best = -1
        history = {'lr': [], 'train_loss': []}

        try:
            print(">> Training %s" % self.model.name)
            for epoch in range(self.args.nepoch):
                with self.experiment.train():
                    train_res = self.train_one_epoch(epoch)

                with self.experiment.validate():
                    print("\nvalidation...")
                    comet_offset = (epoch + 1) * len(self.data['train']) - 1
                    res = self.val(self.data['val'], comet_offset, epoch)

                if res[self.metric] > best:
                    best = res[self.metric]
                    self.save_weights(epoch)

                if self.args.scheduler == 'set':
                    lr = self.optimizer.param_groups[0]['lr']
                    history['lr'].append(lr)
                    history['train_loss'].append(train_res['loss'])

                    self.scheduler.step(epoch + 1)
                    lr = self.optimizer.param_groups[0]['lr']
                    print('learning rate changed to: %.10f' % lr)

                elif self.args.scheduler == 'auto':
                    self.scheduler.step(train_res['loss'])
        finally:
            print(">> Training model %s. [Stopped]" % self.model.name)
            self.experiment.log_asset_folder(os.path.join(self.args.outf, self.args.name, 'weights'),
                                             step=None, log_file_name=False, recursive=False)
            if self.args.scheduler == 'set':
                plt.semilogx(history['lr'], history['train_loss'])
                plt.grid(True)
                self.experiment.log_figure(figure=plt)
                plt.show()

    def val(self, val_loader, comet_offset=-1, epoch=-1):
        self.model.eval()
        test_loss = 0
        correct = 0

        labels = list(range(self.args.nclass))
        cm = np.zeros((len(labels), len(labels)))

        with torch.no_grad():
            for data, target in tqdm(val_loader, leave=True, total=len(val_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                pred = pred.view_as(target).data.cpu().numpy()
                target = target.data.cpu().numpy()
                cm += confusion_matrix(target, pred, labels=labels)

        test_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)

        print('Evaluation: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(val_loader.dataset), accuracy))

        res = {'loss': test_loss, 'acc': accuracy}

        self.experiment.log_metrics(res, step=comet_offset, epoch=epoch)
        self.experiment.log_confusion_matrix(
            matrix=cm,
            labels=[ClassDict.getName(x) for x in labels],
            title='confusion matrix after epoch %03d' % epoch,
            file_name="confusion_matrix_%03d.json" % epoch)

        return res

    def test(self):
        self.load_weights()
        with self.experiment.test():
            print('\ntesting....')
            res = self.val(self.data['test'])

    def log_cmd(self):
        d = vars(self.args)
        cmd = '!python main.py \\\n'
        tab = '    '

        for k, v in d.items():
            if v is None or v == '' or (isinstance(v, bool) and v is False):
                continue

            if isinstance(v, bool):
                arg = '--{} \\\n'.format(k)
            else:
                arg = '--{} {} \\\n'.format(k, v)

            cmd = cmd + tab + arg

        # print(cmd);
        self.experiment.log_text(cmd)

    def save_weights(self, epoch: int):

        weight_dir = os.path.join(self.args.outf, self.args.name, 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()},
                   os.path.join(weight_dir, 'model.pth'))

    def load_weights(self):

        path_g = self.args.weights_path

        if path_g is None:
            weight_dir = os.path.join(self.args.outf, self.args.name, 'weights')
            path_g = os.path.join(weight_dir, 'model.pth')

        print('>> Loading weights...')
        weights_g = torch.load(path_g, map_location=self.device)['state_dict']
        self.model.load_state_dict(weights_g)
        print('   Done.')

    def predict(self, x):
        x = x / 2 ** 15
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = self.transform(x)
            x = x.unsqueeze(0)
            x = self.model(x)
            x = F.softmax(x, dim=1)
            x = x.numpy()
        return x
