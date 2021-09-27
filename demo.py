import argparse
import logging
import torch
import yaml
from matplotlib import pyplot as plt
from dataLoader import createDataset_300W_LP
from model import AmerModel
from transformations import RandomRotation, RandomChoice, VerticalFlip, HorizontalFlip, CenterCrop

################################################

def main():
    pretrained = config['model']['pretrained']
    model_path = config['model']['model_path']
    dev = torch.device('cpu')
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
    print("#Train Batches:", len(train_dataloader),
          "#Test Batches:", len(test_dataloader),
          "#Eval Batches:", len(eval_dataloader))

    ################################################

    if pretrained:
        print("-> Load trained model...")
        try:
            model = torch.load(model_path, map_location=dev)
        except Exception as e:
            print(e)
            raise ValueError('Model not found!')
    else:
        print("-> Create New Model...")
        model = AmerModel()

    if torch.cuda.is_available():
        model.to(dev)

    ################################################

    for i in range(0, len(test_dataloader)):

        with torch.no_grad():
            iterator = iter(test_dataloader)

            image, truth, heat = iterator.next()
            image, truth, heat = image.to(dev), truth.to(dev), heat.to(dev)

            heat_pred, label_pred = model(image)

            for i in range(0, heat_pred.shape[0]):
                image1 = image[i]
                image1 = image1.swapaxes(0, 1)
                image1 = image1.swapaxes(1, 2)
                heat1 = heat[i]
                heat_pred1 = heat_pred[i]

                plt.imshow(image1)
                plt.imshow(heat1, alpha=0.3, cmap="Greys")
                plt.imshow(heat_pred1, alpha=0.3, cmap="Reds")
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--config', type=str, default="config.yml", help="path to the config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    main()
