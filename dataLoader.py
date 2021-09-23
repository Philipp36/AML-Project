import ast
import random
import os
import csv
import sys

import PIL.Image
from skimage import io, transform
import cv2
import numpy
import numpy as np
import pydicom
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pydicom.pixel_data_handlers import apply_voi_lut
from torch.utils.data import DataLoader
import glob
from PIL import Image
################################################
from helperFunctions import *


class DataSetTrain():
    def __init__(self, trainPaths, boundingBoxes, truths, transforms=None):
        self.trainPaths = trainPaths
        self.boundingBoxes = boundingBoxes
        self.truths = truths
        self.transforms = transforms
        self.resize = (448, 448)

    def __getitem__(self, index):
        truth = self.truths[index]
        boxes = self.boundingBoxes[index]

        # Load image
        image = cv2.imread(self.trainPaths[index])
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        img_orig_w, img_orig_h = image.size
        heat = getHeatMap(img_orig_w, img_orig_h, boxes)
        # Resize Image + Box
        plot(image, heat, boxes, True)
        image = image.resize(size=self.resize, resample=PIL.Image.BICUBIC)
        heat = heat.resize(size=self.resize, resample=PIL.Image.BICUBIC)
        img_res_w, img_res_h = image.size
        # Augmentate Image and Box
        trans = torchvision.transforms.PILToTensor()
        image = trans(image)
        heat = trans(heat)
        if self.transforms:
            # fig, axs = plt.subplots(1, 2)
            image, box = self.transforms(image, heat)
            # plot(image, box, axs[0], truth=True)
            # plot(image_da, box_da, axs[1], truth=True)
            # fig.show()
        box = np.array(box)
        return image, int(truth), heat

    def __len__(self):
        return len(self.trainPaths)

################################################


def plot(image, heat, boxes, truth):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    color = 'red' if truth else 'blue'
    ax.imshow(image)
    ax.imshow(heat, alpha=0.3, cmap='hot')
    for box in boxes:
        sx, sy, ex, ey = box
        rect = Rectangle((sx, sy), width=abs(sx - ex), height=abs(sy - ey), fill=None, edgecolor=color)
        ax.add_patch(rect)
    fig.show()


class DataSetTest():
    def __init__(self, trainPaths):
        self.trainPaths = trainPaths

    def __getitem__(self, index):
        image_orig = np.float32(dicom2array(self.trainPaths[index]))

        image = cv2.resize(image_orig, (448, 448))
        return image, image_orig

    def __len__(self):
        return len(self.trainPaths)

################################################


def createDataset_300W_LP(dataset_train_path, dataset_test_path, metadata_image_path, metadata_study_path, data_size=1.0,
                          train_batch=16, test_batch=128, split=0.9, conf_data_loading_train={},
                          conf_data_loading_test={}, transforms=None):
    print("-> Load Data...")

    reader = csv.reader(open(metadata_image_path, 'r'))
    id=[]
    boxes=[]
    label=[]
    studyID=[]
    counter = 0
    for row in reader:
        if counter !=0:
            id.append(row[0])
            boxes.append(row[1])
            label.append(row[2])
            studyID.append(row[3])
        counter +=1

    reader = csv.reader(open(metadata_study_path, 'r'))
    counter = 0

    id_label=[]
    trueLabels=[]
    for row in reader:
        if counter !=0:
            for i in range(1, len(row)):
                if int(row[i])==1:
                    trueLabels.append(i-1)
                    id_label.append(str(os.path.basename(row[0])).replace("_study", ""))
        counter +=1


    testPaths=[]
    testImages = []
    # for folder in os.scandir(dataset_test_path):
    #    for subfolder in os.scandir(folder):
    #        for file in os.scandir(subfolder):
    for path in glob.glob(dataset_test_path + "/**/*.png", recursive=True):
        testPaths.append(path)
        testImages.append(str(os.path.basename(path)).replace(".png", "_image"))

    trainPaths = []
    trainImages = []
    boundingBoxes = []
    truths = []
    counter = 0
    for path in glob.glob(dataset_train_path + "/**/*.png", recursive=True):
    # for folder in os.scandir(dataset_train_path):
    #    for subfolder in os.scandir(folder):
    #         for file in os.scandir(subfolder):
        # path = file.path
        trainPaths.append(path)
        image = str(os.path.basename(path)).replace(".png", "_image")
        trainImages.append(image)
        INDEX = id.index(image)
        box = boxes[INDEX]
        box = getCoordinatesFromBox(box)
        stud = studyID[INDEX]
        INDEX2 = id_label.index(stud)
        truth = trueLabels[INDEX2]
        boundingBoxes.append(box)
        truths.append(truth)

    # Split training set into train and test
    separation = int(len(trainPaths) * split)
    # For augmentation
    val_data = (trainPaths[separation:], boundingBoxes[separation:], truths[separation:])
    # trainPaths, boundingBoxes, truths, types = addAugmentation(trainPaths[:separation], boundingBoxes[:separation], truths[:separation])

    # Shuffle the data
    c = list(zip(trainPaths, boundingBoxes, truths))
    random.shuffle(c)
    trainPaths, boundingBoxes, truths = zip(*c)

    # If we do not want to use 100% of the data
    lim = int(len(trainPaths) * data_size)
    trainPaths = trainPaths[:lim]
    boundingBoxes = boundingBoxes[:lim]
    truths = truths[:lim]

    testPaths = testPaths[:int(len(testPaths) * data_size)]


    # For training
    dataset_Train = DataSetTrain(trainPaths, boundingBoxes, truths, transforms=transforms)

    dataloader_Train = DataLoader(dataset=dataset_Train, batch_size=train_batch, shuffle=True,
                                  drop_last=True, **conf_data_loading_train)

    # For testing
    dataset_Test = DataSetTrain(*val_data)
    dataloader_Test = DataLoader(dataset=dataset_Test, batch_size=test_batch, shuffle=True,
                                 drop_last=True, **conf_data_loading_test)

    # For actual evaluation set
    dataset_Test_Eval = DataSetTest(testPaths)
    dataloader_Test_Eval = DataLoader(dataset=dataset_Test_Eval, batch_size=test_batch, shuffle=False, drop_last=True)

    return dataloader_Train, dataloader_Test, dataloader_Test_Eval


def getCoordinatesFromBox(box):
    try:
        box = ast.literal_eval(box)
        boxes = []
        for b in box:
            boxes.append([b["x"], b["y"], b["x"]+b["width"], b["y"]+b["height"]])
        return np.array(boxes)
    except:
        return np.array([[0, 0, 1, 1]])


def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)