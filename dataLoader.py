import ast
import random
import os
import csv
import sys
from skimage import io, transform
import cv2
import numpy
import numpy as np
import pydicom
import torch
import torchvision
from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers import apply_voi_lut
from torch.utils.data import DataLoader
import glob
from PIL import Image
################################################
from helperFunctions import *


class DataSetTrain():
    def __init__(self, trainPaths, boundingBoxes, truths, types, demo):
        self.trainPaths = trainPaths
        self.boundingBoxes = boundingBoxes
        self.truths = truths
        self.demo = demo
        self.types = types

    def __getitem__(self, index):
        truth = self.truths[index]

        image_orig = cv2.imread(self.trainPaths[index])
        image_orig = Image.fromarray(image_orig.astype('uint8'), 'RGB')
        box = self.boundingBoxes[index]

        if self.types[index] == 1:
            image_orig, box = mirrorImage(image_orig, box)
        elif self.types[index] == 2:
            image_orig, box = rotateImage(image_orig, box, option="left")
        elif self.types[index] == 3:
            image_orig, box = rotateImage(image_orig, box, option="right")
        image_orig = np.array(image_orig)


        image = cv2.resize(image_orig, (448, 448))
        heat_orig = getHeatMap(torch.tensor(image_orig), torch.tensor(box))
        sx, sy, ex, ey = box
        x, y, _ = image.shape
        x1, y1, _ = image_orig.shape
        scaleX = x / x1
        scaleY = y / y1
        sx = sx * scaleY
        sy = sy * scaleX
        ex = ex * scaleY
        ey = ey * scaleX
        box_pred_origsize = np.array([sx, sy, ex, ey])
        heat_resized = getHeatMap(torch.tensor(image), torch.tensor(numpy.array(box_pred_origsize)))

        if self.demo:
            return np.moveaxis(image, -1, 0), heat_resized, int(truth), box_pred_origsize.astype(np.float32, copy=False), np.moveaxis(image_orig, -1, 0), heat_orig

        else:
            return np.moveaxis(image, -1, 0), heat_resized, int(truth), box_pred_origsize.astype(np.float32, copy=False)


    def __len__(self):
        return len(self.trainPaths)

################################################


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
                          train_batch=16, test_batch=128, split=0.9, demo=False, conf_data_loading={}):
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
    trainPaths, boundingBoxes, truths, types = addAugmentation(trainPaths[:separation], boundingBoxes[:separation], truths[:separation])

    # Shuffle the data
    c = list(zip(trainPaths, boundingBoxes, truths, types))
    random.shuffle(c)
    trainPaths, boundingBoxes, truths, types = zip(*c)

    # If we do not want to use 100% of the data
    lim = int(len(trainPaths) * data_size)
    trainPaths = trainPaths[:lim]
    boundingBoxes = boundingBoxes[:lim]
    truths = truths[:lim]
    types = types[:lim]

    testPaths = testPaths[:int(len(testPaths) * data_size)]


    # For training
    dataset_Train = DataSetTrain(trainPaths, boundingBoxes, truths, types, demo=demo)

    dataloader_Train = DataLoader(dataset=dataset_Train, batch_size=train_batch, shuffle=True,
                                  drop_last=True, **conf_data_loading)

    # For testing
    dataset_Test = DataSetTrain(*val_data, np.zeros(shape=(len(val_data[0]), 1)), demo=demo)
    dataloader_Test = DataLoader(dataset=dataset_Test, batch_size=test_batch, shuffle=True,
                                 drop_last=True, **conf_data_loading)

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
        return np.array(boxes)[0]
    except:
        return np.array([0, 0, 1, 1])

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