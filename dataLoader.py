import ast
import random
import os
import csv
import cv2
import numpy
import numpy as np
import pydicom
import torch
from pydicom.pixel_data_handlers import apply_voi_lut
from torch.utils.data import DataLoader
import glob
################################################
from helperFunctions import getHeatMap, rescaleBox


class DataSetTrain():
    def __init__(self, trainPaths, boundingBoxes, truths, demo):
        self.trainPaths = trainPaths
        self.boundingBoxes = boundingBoxes
        self.truths = truths
        self.demo = demo

    def __getitem__(self, index):


        truth = self.truths[index]
        # image_orig = np.float32(dicom2array(self.trainPaths[index]))
        image_orig = cv2.imread(self.trainPaths[index])
        image = cv2.resize(image_orig, (448, 448))
        image = image.astype(np.float32, copy=False)
        # image = torch.from_numpy(image)
        box = self.boundingBoxes[index]
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
        box_pred_origsize = [sx, sy, ex, ey]
        heat_resized = getHeatMap(torch.tensor(image), torch.tensor(numpy.array(box_pred_origsize)))
        if self.demo:
            return image, image_orig, heat_resized, heat_orig, int(truth), box.astype(np.float32, copy=False)
        else:
            return np.moveaxis(image, -1, 0), heat_resized, int(truth), box.astype(np.float32, copy=False)


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
                          BATCH=64, split=0.9, demo=False, conf_data_loading={}):
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
    for folder in os.scandir(dataset_test_path):
        for subfolder in os.scandir(folder):
            for file in os.scandir(subfolder):
                testPaths.append(file.path)
                testImages.append(str(os.path.basename(file)).replace(".png", "_image"))

    trainPaths = []
    trainImages = []
    boundingBoxes = []
    truths = []
    counter = 0
#    for folder in os.scandir(dataset_train_path):
#        for subfolder in os.scandir(folder):
#            for file in os.scandir(subfolder):
    for path in glob.glob(dataset_train_path + "/**/*.png", recursive=True):
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

    # Split training set into train and test
    separation = int(len(trainPaths) * split)

    # For training
    dataset_Train = DataSetTrain(trainPaths[:separation], boundingBoxes[:separation], truths[:separation], demo=demo)

    dataloader_Train = DataLoader(dataset=dataset_Train, batch_size=BATCH, shuffle=True,
                                  drop_last=True, **conf_data_loading)

    # For testing
    dataset_Test = DataSetTrain(trainPaths[separation:], boundingBoxes[separation:], truths[separation:], demo=demo)
    dataloader_Test = DataLoader(dataset=dataset_Test, batch_size=BATCH, shuffle=True,
                                 drop_last=True, **conf_data_loading)

    # For actual evaluation set
    dataset_Test_Eval = DataSetTest(testPaths)
    dataloader_Test_Eval = DataLoader(dataset=dataset_Test_Eval, batch_size=BATCH, shuffle=False, drop_last=True)

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
