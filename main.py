from comet_ml import Experiment
from src.configs import get_args
from src.data import get_dataloader
from src.trainer import ModelTrainer
from src.model import *


def load_model(args):
    model_constructor = globals()[args.model];
    model = model_constructor(args);
    return model;


def run():

    args = get_args();

    dataloader = get_dataloader(args);

    model = load_model(args);

    trainer = ModelTrainer(model, dataloader, args);

    trainer.train() if not args.test else trainer.test();


if __name__ == '__main__':
    run();
    print('done');
    exit();

