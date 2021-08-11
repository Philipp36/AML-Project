import argparse
import numpy as np
import torch
from dataLoader import createDataset_300W_LP
from helperFunctions import *
from model import model1
################################################

def main(args):

    modelType = args.modelType

    train_dataloader, test_dataloader, eval_dataloader =  createDataset_300W_LP(dataSize=0.1, BATCH=1, split=0.8, demo = True)
    print("#Train Batches:", len(train_dataloader.dataset), "#Test Batches:", len(test_dataloader.dataset), "#Eval Batches:", len(eval_dataloader.dataset))
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

    with torch.no_grad():
        iterator = iter(test_dataloader)
        image, box, truth, imageOrig = iterator.next()

        box_pred, label_pred = model(image)

        image = image[0]
        box = box[0]
        box_pred = box_pred[0]
        imageOrig = imageOrig[0]
        label_pred = label_pred[0]
        truth = truth[0]

        print("True Bounding Box:  ", np.array(box))
        print("Predicted Bounding Box:  ", np.array(box_pred))
        print("True Label:  ", int(truth))
        print("Predicted Label:  ", int(torch.argmax(label_pred)))

        # You can decide wheter to plot the original size image, or the resized
        drawSample(image, imageOrig, box, box_pred, originalSize = True)

################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--modelType', type=str, default="NEW", help="NEW or TRAINED")
    args = parser.parse_args()
    main(args)