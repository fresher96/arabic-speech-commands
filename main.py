from src.configs import get_args
from src.data import get_dataloader
from src.trainer import ModelTrainer
from src.model import *


def run():

    args = get_args();

    dataloader = get_dataloader(args);

    model = CompressModel(args);

    trainer = ModelTrainer(model, dataloader, args);

    if(args.test):
        trainer.test();
    else:
        trainer.train();
        trainer.test();





if __name__ == '__main__':
    run();
    print('done');
    exit();
