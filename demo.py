import matplotlib.pyplot as plt
import torch
from dataLoader import createDataset_300W_LP
import argparse

################################################

def main(args):

    modelType = args.modelType

################################################
    train_dataloader, test_dataloader, eval_dataloader =  createDataset_300W_LP(dataSize=0.1, BATCH=1, split=0.8)
    print("#Train Batches:", len(train_dataloader.dataset), "#Test Batches:", len(test_dataloader.dataset), "#Eval Batches:", len(eval_dataloader.dataset))
################################################

    if modelType == "TRAINED":
        print("-> Load trained model...")
        try:
            #Load model here
            a = 0
        except Exception as e:
            print(e)
            raise ValueError('Model not found!')
    else:
        print("-> Create New Model...")
        # Create new model here
        a = 0

################################################

    with torch.no_grad():
        iterator = iter(test_dataloader)
        image, box, truth = iterator.next()
        print("True Bounding Box:  ", box)
        print("True Label:  ", truth)
        plt.imshow(image[0])
        plt.show()

        # TODO:
        # We reshape our images to forward them all in the same shape. e.g 128x128
        # We then predict bounding box coordinates on the reshaped images
        # Therefore, we need a method, that retransformes the bounding box coordinates
        #into the scale of the original image

################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--modelType', type=str, default="NEW", help="NEW or TRAINED")
    args = parser.parse_args()
    main(args)