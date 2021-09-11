import argparse
import sys
import yaml

import numpy as np
import torch
import logging
from dataLoader import createDataset_300W_LP
from helperFunctions import *
from model import model1, AmerModel


################################################


def main():
    pretrained = config['model']['pretrained']
    model_path = config['model']['model_path']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(torch.cuda.is_available())
    torch.cuda.empty_cache()
    train_dataloader, test_dataloader, eval_dataloader = createDataset_300W_LP(**config['dataset'],
                                                                               BATCH=config['train']['batch_size'],
                                                                               conf_data_loading=config['data_loading'])
    print("#Train Batches:", len(train_dataloader), "#Test Batches:", len(test_dataloader), "#Eval Batches:",
          len(eval_dataloader))

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

    ################################################

################################################

    for i in range(0, len(test_dataloader)):

        with torch.no_grad():
            iterator = iter(test_dataloader)

            image, truth_heatMap_resized, truth, box, imageOrig, truth_heatMap_origsize = iterator.next()
            #image, heat_resized, truth, box = iterator.next()

            heat_pred, label_pred = model(image)

            for i in range(0, heat_pred.shape[0]):
                #print(truth[i], torch.argmax(label_pred[i]))
                #print(box[i])
                #print(heat_pred[i])
                #print("")

                image = image[i]
                image = image.swapaxes(0, 1)
                image = image.swapaxes(1, 2)
                imageOrig = imageOrig[i]
                imageOrig = imageOrig.swapaxes(0, 1)
                imageOrig = imageOrig.swapaxes(1, 2)

                drawSample(image, imageOrig, box[i], heat_pred[i], originalSize=False)



            """
            image = image[0]
            image = image.swapaxes(0, 1)
            image = image.swapaxes(1, 2)
            imageOrig = imageOrig[0]
            imageOrig = imageOrig.swapaxes(0, 1)
            imageOrig = imageOrig.swapaxes(1, 2)
            heat_pred = heat_pred.reshape((-1, 448, 448))
            box_pred = getBoxFromHeatMap(heat_pred)
            box = box[0]
            box_pred = box_pred[0]
            label_pred = label_pred[0]
            truth = truth[0]
            truth_heatMap_resized = truth_heatMap_resized[0]
            truth_heatMap_origsize = truth_heatMap_origsize[0]
            print("True Label:  ", int(truth))
            print("Predicted Label:  ", int(torch.argmax(label_pred)))
            # You can decide wheter to plot the original size image, or the resized
            drawSample(image, imageOrig, box, box_pred, originalSize = True)
            #drawSample(truth_heatMap_resized, truth_heatMap_origsize, box, box_pred, originalSize=False)
            """
################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--config', type=str, default="config.yml", help="path to the config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main()