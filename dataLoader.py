import ast
import random
import os
import csv
import sys
import cv2
import numpy as np
import pydicom
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
################################################

class DataSetTrain():
    def __init__(self, trainPaths, boundingBoxes, truths):
        self.trainPaths = trainPaths
        self.boundingBoxes = boundingBoxes
        self.truths = truths
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128)])

    def __getitem__(self, index):

        box = self.boundingBoxes[index]
        truth = self.truths[index]

        image_file = pydicom.dcmread(self.trainPaths[index])
        image = np.array(image_file.pixel_array, dtype=np.float32)[np.newaxis]
        image = torch.from_numpy(image)
        image = self.transform(image)
        image = torch.squeeze(image)

        return image, box, int(truth)

    def __len__(self):
        return len(self.trainPaths)

################################################

class DataSetTest():
    def __init__(self, trainPaths):
        self.trainPaths = trainPaths

    def __getitem__(self, index):
        image_file = pydicom.dcmread(self.trainPaths[index])
        image = np.array(image_file.pixel_array, dtype=np.float32)[np.newaxis]
        image = torch.from_numpy(image)
        image = self.transform(image)
        image = torch.squeeze(image)
        return image

    def __len__(self):
        return len(self.trainPaths)

################################################

def createDataset_300W_LP(dataSize=1.0, BATCH=64, split=0.9):
    print("-> Load Data...")

    pathCSV = "data/train_image_level.csv"
    reader = csv.reader(open(pathCSV, 'r'))
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

    pathCSV = "data/train_study_level.csv"
    reader = csv.reader(open(pathCSV, 'r'))
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
    testDir = "data/test"
    for folder in os.scandir(testDir):
        for subfolder in os.scandir(folder):
            for file in os.scandir(subfolder):
                testPaths.append(file.path)
                testImages.append(str(os.path.basename(file)).replace(".dcm", "_image"))

    trainPaths = []
    trainImages = []
    boundingBoxes = []
    truths = []
    testDir = "data/train"
    for folder in os.scandir(testDir):
        for subfolder in os.scandir(folder):

            for file in os.scandir(subfolder):
                trainPaths.append(file.path)
                image = str(os.path.basename(file)).replace(".dcm", "_image")
                trainImages.append(image)
                INDEX = id.index(image)
                box = boxes[INDEX]
                box = getCoordinatesFromBox(box)
                stud = studyID[INDEX]
                INDEX2 = id_label.index(stud)
                truth = trueLabels[INDEX2]
                boundingBoxes.append(box)
                truths.append(truth)

    # Shuffle the data
    c = list(zip(trainPaths, boundingBoxes, truths))
    random.shuffle(c)
    trainPaths, boundingBoxes, truths = zip(*c)

    # If we do not want to use 100% of the data
    lim = int(len(trainPaths)*dataSize)
    trainPaths = trainPaths[:lim]
    boundingBoxes = boundingBoxes[:lim]
    truths = truths[:lim]
    testPaths = testPaths[:int(len(testPaths)*dataSize)]

    # Split training set into train and test
    separation = int(len(trainPaths) * split)

    # For training
    dataset_Train = DataSetTrain(trainPaths[:separation], boundingBoxes[:separation], truths[:separation])
    dataloader_Train = DataLoader(dataset=dataset_Train, batch_size=BATCH, shuffle=True)

    # For testing
    dataset_Test = DataSetTrain(trainPaths[separation:], boundingBoxes[separation:], truths[separation:])
    dataloader_Test = DataLoader(dataset=dataset_Test, batch_size=BATCH, shuffle=True)

    # For actual evaluation set
    dataset_Test_Eval = DataSetTest(testPaths)
    dataloader_Test_Eval = DataLoader(dataset=dataset_Test_Eval, batch_size=BATCH, shuffle=True)

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

