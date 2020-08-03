import os;
from comet_ml import Experiment
import torch;
from torch import nn;
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
import numpy as np;
import random;
from sklearn.metrics import confusion_matrix

from src.ClassDict import ClassDict;

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


        self.experiment = Experiment(api_key=args.comet_key, auto_weight_logging=True,
                                     project_name=args.comet_project, workspace=args.comet_workspace)

        self.experiment.set_name(args.name);
        self.experiment.log_parameters(vars(args));
        self.experiment.set_model_graph(str(self.model));

    def train_one_epoch(self, epoch):

        self.model.train()
        train_loader = self.data['train'];
        train_loss = 0
        correct = 0

        comet_offset = epoch * len(train_loader);

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

            loss = loss.item() / len(data);
            acc = 100. * acc / len(data);

            comet_step = comet_offset + batch_idx;
            self.experiment.log_metric('batch_loss', loss, comet_step, epoch);
            self.experiment.log_metric('batch_acc', acc, comet_step, epoch);

            if (batch_idx + 1) % self.args.frq_log == 0:
                self.experiment.log_metric('log_loss', loss, comet_step, epoch);
                self.experiment.log_metric('log_acc', acc, comet_step, epoch);
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                    epoch + 1, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss, acc))

        train_loss /= len(train_loader.dataset)
        acc = 100. * correct / len(train_loader.dataset);

        comet_step = comet_offset + len(train_loader) - 1;
        self.experiment.log_metric('loss', train_loss, comet_step, epoch);
        self.experiment.log_metric('acc', acc, comet_step, epoch);

        print('Epoch: {} [Done]\tLoss: {:.4f}\tAccuracy: {}/{} ({:.2f}%)'.format(
            epoch + 1, train_loss, correct, len(train_loader.dataset), acc))


    def train(self):

        best = 0

        print(">> Training %s" % self.model.name)
        for epoch in range(self.args.nepoch):
            with self.experiment.train():
                self.train_one_epoch(epoch)

            with self.experiment.validate():
                print("\nvalidation...");
                comet_offset = (epoch + 1) * len(self.data['train']) - 1;
                res = self.val(self.data['val'], comet_offset, epoch)

            if res[self.metric] > best:
                best = res[self.metric]
                # self.save_weights(self.epoch)

        print(">> Training model %s.[Done]" % self.model.name)

    def val(self, val_loader, comet_offset=-1, epoch=-1):
        self.model.eval()
        test_loss = 0
        correct = 0

        test_targets = None;
        test_predictions = None;

        # cm = cm.compute_matrix(y_true, y_predicted)

        with torch.no_grad():
            for data, target in tqdm(val_loader, leave=True, total=len(val_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                pred = pred.view_as(target).data.numpy();
                test_predictions = pred if test_predictions is None else np.concatenate((test_predictions, pred));

                target = target.data.numpy();
                test_targets = target if test_targets is None else np.concatenate((test_targets, target));

        test_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset);

        print('Evaluation: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(val_loader.dataset), accuracy))

        res = {'loss': test_loss, 'acc': accuracy};
        self.experiment.log_metrics(res, step=comet_offset, epoch=epoch);

        # test_targets = list(test_targets);
        matrix = confusion_matrix(test_targets, test_predictions)

        self.experiment.log_confusion_matrix(
            matrix=matrix,
            title='confusion matrix after epoch %03d' % epoch,
            file_name="confusion_matrix_%03d.json" % epoch,
            # labels=ClassDict.dct,
        )

        return res;

    def test(self):
        # load_wights();
        with self.experiment.test():
            print('\ntesting....');
            res = self.val(self.data['test']);
