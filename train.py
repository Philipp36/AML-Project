import argparse
import numpy as np
import torch
from torch import optim, nn
from tqdm import trange

from dataLoader import createDataset_300W_LP
from helperFunctions import *
from model import model1
from torch.utils.tensorboard import SummaryWriter
################################################

def main(args):

    modelType = args.modelType
    pathModel = "weights/model.pth"

    train_dataloader, test_dataloader, eval_dataloader =  createDataset_300W_LP(dataSize=1, BATCH=32, split=0.8, demo = False)
    print("#Train Batches:", len(train_dataloader), "#Test Batches:", len(test_dataloader), "#Eval Batches:", len(eval_dataloader))
################################################

    if modelType == "TRAINED":
        print("-> Load trained model...")
        try:
            model = torch.load("weights/model.pth")
        except Exception as e:
            print(e)
            raise ValueError('Model not found!')
    else:
        print("-> Create New Model...")
        model = model1()

################################################
    epochs = 5
    optimizer = optim.Adam(model.parameters())

    writer = SummaryWriter()
    counter_test = 0

    print("")
    trainLoop(model, optimizer, epochs, train_dataloader, test_dataloader, counter_test, writer, pathModel)
    print("")

    print("-> Save trained model...")
    torch.save(model, pathModel)
################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--modelType', type=str, default="NEW", help="NEW or TRAINED")
    args = parser.parse_args()
    main(args)