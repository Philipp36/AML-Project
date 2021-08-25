import argparse
import sys

import numpy as np
import torch
from torch import optim, nn
from tqdm import trange
import pytorch_lightning as pl
from dataLoader import createDataset_300W_LP
from helperFunctions import *
from model import AmerModel
from torch.utils.tensorboard import SummaryWriter


import yaml
################################################


def main():
    pretrained = config['model']['pretrained']
    model_path = config['model']['model_path']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, test_dataloader, eval_dataloader = createDataset_300W_LP(**config['dataset'],
                                                                               BATCH=config['train']['batch_size'])
    print("#Train Batches:", len(train_dataloader), "#Test Batches:", len(test_dataloader), "#Eval Batches:", len(eval_dataloader))
################################################

    if pretrained:
        print("-> Load trained model...")
        try:
            model = torch.load(model_path)
        except Exception as e:
            print(e)
            raise ValueError('Model not found!')
    else:
        print("-> Create New Model...")
        model = AmerModel()

    model.to(dev)

################################################
    epochs = config['train']['epochs']

    print("")
    modelTrainer = trainerLightning(model, config, dev = dev)
    trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=1 , profiler="advanced")
    trainer.fit(modelTrainer, train_dataloader, test_dataloader)
    print("")

    print("-> Save trained model...")
    torch.save(model, model_path)
################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--config', type=str, default="config.yml", help="path to the config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main()