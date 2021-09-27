import ast
import pickle
import csv
import glob
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

import pydicom
import cv2
from PIL import Image
from pydicom.pixel_data_handlers import apply_voi_lut

from helperFunctions import getHeatMap


class DataSetTrain(Dataset):
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
        if np.array_equal(boxes, np.array([[0, 0, 1, 1]])):
            heat = np.zeros((img_orig_h, img_orig_w), dtype=np.float32)
        else:
            heat = getHeatMap(img_orig_w, img_orig_h, boxes)
        # Resize Image + Box
        image = image.resize(size=self.resize, resample=Image.BICUBIC)
        heat = cv2.resize(heat, self.resize)
        heat = torch.from_numpy(heat)
        heat = torch.unsqueeze(heat, 0)
        # Augmentate Image and Box
        image = pil_to_tensor(image)
        if self.transforms:
            image, heat = self.transforms(image, heat)
        heat = torch.squeeze(heat)
        return image, int(truth), heat

    def __len__(self):
        return len(self.trainPaths)


class DataSetTest(Dataset):
    def __init__(self, trainPaths):
        self.trainPaths = trainPaths

    def __getitem__(self, index):
        image_orig = np.float32(dicom2array(self.trainPaths[index]))

        image = cv2.resize(image_orig, (448, 448))
        return image, image_orig

    def __len__(self):
        return len(self.trainPaths)


def createDataset_300W_LP(dataset_train_path, dataset_test_path, metadata_image_path, metadata_study_path, data_size=1.0,
                          train_batch=16, test_batch=128, split=0.9, conf_data_loading_train={},
                          conf_data_loading_test={}, transforms=None):
    print("-> Load Data...")

    reader = csv.reader(open(metadata_image_path, 'r'))
    id = []
    boxes = []
    label = []
    studyID = []
    counter = 0
    for row in reader:
        if counter != 0:
            id.append(row[0])
            boxes.append(row[1])
            label.append(row[2])
            studyID.append(row[3])
        counter += 1

    reader = csv.reader(open(metadata_study_path, 'r'))
    counter = 0

    id_label = []
    trueLabels = []
    for row in reader:
        if counter !=0:
            for i in range(1, len(row)):
                if int(row[i]) == 1:
                    trueLabels.append(i-1)
                    id_label.append(str(os.path.basename(row[0])).replace("_study", ""))
        counter += 1


    testPaths=[]
    testImages = []

    for path in glob.glob(dataset_test_path + "/**/*.png", recursive=True):
        testPaths.append(path)
        testImages.append(str(os.path.basename(path)).replace(".png", "_image"))

    trainPaths = []
    trainImages = []
    boundingBoxes = []
    truths = []
    counter = 0
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

    open_file = open("trainPaths.pkl", "rb")
    relPaths = pickle.load(open_file)
    open_file.close()
    # JOIN BASE_PATH TO
    trainPaths = [os.path.join(dataset_train_path, rel_path) for rel_path in relPaths]

    open_file = open("boundingBoxes.pkl", "rb")
    boundingBoxes = pickle.load(open_file)
    open_file.close()

    open_file = open("truths.pkl", "rb")
    truths = pickle.load(open_file)
    open_file.close()
    ###

    separation = int(len(trainPaths) * 0.8)
    val_data = (trainPaths[separation:], boundingBoxes[separation:], truths[separation:])
    trainPaths, boundingBoxes, truths = trainPaths[:separation], boundingBoxes[:separation], truths[:separation]


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
