import argparse
import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import helperFunctions
from dataLoader import createDataset_300W_LP
from helperFunctions import *
from model import AmerModel
from transformations import RandomRotation, VerticalFlip, HorizontalFlip, CenterCrop, RandomChoice
from lightningTrainer import trainerLightning
import yaml
from torch.utils.tensorboard import SummaryWriter
################################################


def main():
    pretrained = config['model']['pretrained']
    model_path = config['model']['model_path']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    transforms = RandomChoice(transforms=[
        RandomRotation(degrees=(-90, 90)),
        VerticalFlip(),
        HorizontalFlip(),
        CenterCrop(size=(425, 425))
    ], p=0.4)
    train_dataloader, test_dataloader, eval_dataloader = \
        createDataset_300W_LP(**config['dataset'], train_batch=config['train']['batch_size'],
                              test_batch=config['test']['batch_size'],
                              conf_data_loading_train=config['train']['data_loading'],
                              conf_data_loading_test=config['test']['data_loading'], transforms=transforms)

    model_name = 'resnet-18-train-all-heatmaps'
    base_log_folder = 'logs2'
    log_folder = os.path.join(base_log_folder, model_name)
    os.makedirs(log_folder)
    with open(os.path.join(log_folder, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    print("#Train Batches:", len(train_dataloader),
          "#Test Batches:", len(test_dataloader),
          "#Eval Batches:", len(eval_dataloader))

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

    if torch.cuda.is_available():
        model.to(dev)
################################################

    epochs = config['train']['epochs']
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    counter_test = 0
    writer = SummaryWriter()

    use_lightning = False
    if use_lightning:
        logger = TensorBoardLogger(save_dir='runs', name='', default_hp_metric=False)
        modelTrainer = trainerLightning(model, config, dev=dev)
        trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=1, log_every_n_steps=len(train_dataloader),
                             logger=logger, profiler="simple", gpus=1 if torch.cuda.is_available() else 0)
        trainer.fit(modelTrainer, train_dataloader, test_dataloader)
    else:
        print("")
        trainLoop(model, optimizer, epochs, train_dataloader, test_dataloader, counter_test, writer, model_path, log_folder,
                  dev=dev)
        print("")

    print("-> Save trained model...")
    torch.save(model, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--config', type=str, default="config.yml", help="path to the config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main()
