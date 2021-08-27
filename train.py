import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from dataLoader import createDataset_300W_LP
from helperFunctions import *
from model import AmerModel
from lightningTrainer import trainerLightning
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
    logger = TensorBoardLogger(save_dir='runs', name='', default_hp_metric=False)
    modelTrainer = trainerLightning(model, config, dev = dev)
    trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=1, log_every_n_steps=len(train_dataloader), logger=logger, profiler="simple")
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