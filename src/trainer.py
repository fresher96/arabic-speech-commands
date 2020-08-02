import os;
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

        self.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum);
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_one_epoch(self, epoch):

        self.model.train()
        train_loader = self.data['train'];
        # print(len(train_loader));

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            # print(target);

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            if (batch_idx + 1) % self.args.frq_log == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader), loss.item()))

        print(">> Epoch %d/%d" % (epoch + 1, self.args.epochs))

    def train(self):

        best = 0

        print(">> Training %s".format(self.model.name))
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)
            # res = self.test()
            # if res['AUC'] > best:
            #     best = res['AUC']
            #     self.save_weights(self.epoch)
        print(">> Training model %s.[Done]" % self.model.name)


    def test(self):

        self.test();
        with torch.no_grad():

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.latent_i = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            self.latent_o = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)

            # print("   Testing model %s." % self.model.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))
                self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(
                    error.size(0), self.opt.nz)
                self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(
                    error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                        real, fake, _ = self.get_current_images()
                        vutils.save_image(real, '%s/real_%03d.eps' % (dst, i + 1), normalize=True)
                        vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i + 1), normalize=True)

                        # Measure inference time.
                        self.times = np.array(self.times)
                        self.times = np.mean(self.times[:100] * 1000)

                        # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

        ##
        def update_learning_rate(self):
            """ Update learning rate based on the rule provided in options.
            """

            for scheduler in self.schedulers:
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]['lr']
            print('   LR = %.7f' % lr)


