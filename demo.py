import argparse
import sys
import yaml

import numpy as np
import torch

from dataLoader import createDataset_300W_LP
from helperFunctions import *
from model import model1
################################################


def main():
    pretrained = False
    model_path = config['model']['model_path']
    train_dataloader, test_dataloader, eval_dataloader = createDataset_300W_LP(**{key: val for key, val in config['dataset'].items() if 'path' in key}, data_size=0.04, BATCH=1, split=0.8,
                                                                               demo=True)
    print("#Train Batches:", len(train_dataloader.dataset), "#Test Batches:", len(test_dataloader.dataset), "#Eval Batches:", len(eval_dataloader.dataset))
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
        model = model1()

################################################

    with torch.no_grad():
        iterator = iter(test_dataloader)
        image, imageOrig, truth_heatMap_resized, truth_heatMap_origsize, truth, box = iterator.next()

        box_pred, label_pred = model(image)
        image = image[0]
        box = box[0]
        box_pred = box_pred[0]
        imageOrig = imageOrig[0]
        label_pred = label_pred[0]
        truth = truth[0]
        truth_heatMap_resized = truth_heatMap_resized[0]
        truth_heatMap_origsize = truth_heatMap_origsize[0]

        print("True Label:  ", int(truth))
        print("Predicted Label:  ", int(torch.argmax(label_pred)))

        # You can decide wheter to plot the original size image, or the resized
        drawSample(image, imageOrig, box, box_pred, originalSize = True)
        #drawSample(truth_heatMap_resized, truth_heatMap_origsize, box, box_pred, originalSize=False)

################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--config', type=str, default="config.yml", help="path to the config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main()
