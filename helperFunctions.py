import sys

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, cm
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
############################################################################
from torch import nn, optim
from tqdm import trange
import cv2

def rescaleBox(box_pred, box_true, img, imageOrig, transformPredicted = False):

    # Transform the predicted Box
    if transformPredicted:
        sx, sy, ex, ey = box_pred
        x, y = img.shape
        x1, y1 = imageOrig.shape
        scaleX = x1/x
        scaleY = y1/y

    # Transform the true Box
    else:
        sx, sy, ex, ey = box_true
        x, y = img.shape
        x1, y1 = imageOrig.shape
        scaleX = x/x1
        scaleY = y/y1

    sx = sx * scaleY
    sy = sy * scaleX
    ex = ex * scaleY
    ey = ey * scaleX

    return torch.tensor([sx, sy, ex, ey])

############################################################################

def drawSample(IMG, IMG_ORIG, box_true, box_pred, originalSize = False):

    if originalSize:
        # Draw real size
        BOX_PREDICTED = rescaleBox(box_pred, box_true, IMG, IMG_ORIG, transformPredicted=True)

        plt.imshow(IMG_ORIG)

        sx, sy, ex, ey = box_true

        plt.plot([sx, sx], [sy, ey], lw=2, c = "red", label = "True")
        plt.plot([sx, ex], [sy, sy], lw=2, c = "red")
        plt.plot([ex, ex], [ey, sy], lw=2, c = "red")
        plt.plot([ex, sx], [ey, ey], lw=2, c = "red")

        sx, sy, ex, ey = BOX_PREDICTED
        plt.plot([sx, sx], [sy, ey], lw=2, c = "blue", label = "Predicted")
        plt.plot([sx, ex], [sy, sy], lw=2, c = "blue")
        plt.plot([ex, ex], [ey, sy], lw=2, c = "blue")
        plt.plot([ex, sx], [ey, ey], lw=2, c = "blue")
        plt.legend()
        plt.show()

    else:
        # Draw resized
        BOX_TRUTH = rescaleBox(box_pred, box_true, IMG, IMG_ORIG, transformPredicted=False)

        sx, sy, ex, ey = BOX_TRUTH

        plt.plot([sx, sx], [sy, ey], lw=2, c = "red", label = "True")
        plt.plot([sx, ex], [sy, sy], lw=2, c = "red")
        plt.plot([ex, ex], [ey, sy], lw=2, c = "red")
        plt.plot([ex, sx], [ey, ey], lw=2, c = "red")

        sx, sy, ex, ey = box_pred
        plt.plot([sx, sx], [sy, ey], lw=2, c = "blue", label = "Predicted")
        plt.plot([sx, ex], [sy, sy], lw=2, c = "blue")
        plt.plot([ex, ex], [ey, sy], lw=2, c = "blue")
        plt.plot([ex, sx], [ey, ey], lw=2, c = "blue")
        plt.legend()

        plt.imshow(IMG)
        plt.show()


############################################################################
def trainLoop(model, optimizer, epochs, train_dataloader, test_dataloader, counter_test, writer, pathModel, dev=torch.device('cpu')):

    testLoop(model, test_dataloader, counter_test, writer, dev=dev)
    counter_test += 1

    lossBoxes = nn.MSELoss()
    lossLabels = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(0, epochs):
        print(" Epoch:  ", epoch)
        iterator = iter(train_dataloader)
        losses1 = []
        losses2 = []
        for index in trange(0, len(train_dataloader)):
            image, heat_resized, truth, box = iterator.next()
            image, heat_resized, truth, box = image.to(dev, non_blocking=True), heat_resized.to(dev, non_blocking=True), truth.to(dev, non_blocking=True), box.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            heat_pred, label_pred = model(image)
            loss1 = lossBoxes(heat_pred.double(), heat_resized.double())
            loss2 = lossLabels(label_pred, truth)
            #if epoch == 0 and index == 0:
            #    W = float((loss1 / loss2))
            LOSS = loss1 + loss2 #(W * loss2)
            LOSS.backward()
            optimizer.step()
            losses1.append(float(loss1))
            losses2.append(float(loss2))
            del LOSS

        loss1 = np.mean(np.array(losses1))
        loss2 = np.mean(np.array(losses2))
        writer.add_scalar('BoxLoss/train/', loss1, epoch)
        writer.add_scalar('LabelLoss/train/', loss2, epoch)
        testLoop(model, test_dataloader, counter_test, writer, dev=dev)
        counter_test += 1
        torch.save(model, pathModel)

############################################################################

def testLoop(model, test_dataloader, counter_test, writer, dev=torch.device('cpu')):
    with torch.no_grad():
        model.eval()

        print("Test ", counter_test)
        lossBoxes = nn.MSELoss()
        lossLabels = nn.CrossEntropyLoss()
        iterator = iter(test_dataloader)
        losses1 = []
        losses2 = []
        for index in range(0, len(test_dataloader)):
            image, heat_resized, truth, box = iterator.next()
            image, heat_resized, truth, box = image.to(dev, non_blocking=True), heat_resized.to(dev, non_blocking=True), truth.to(dev, non_blocking=True), box.to(dev, non_blocking=True)
            heat_pred, label_pred = model(image)
            loss1 = lossBoxes(heat_pred, heat_resized)
            loss2 = lossLabels(label_pred, truth)
            losses1.append(float(loss1))
            losses2.append(float(loss2))

        loss1 = np.mean(np.array(losses1))
        loss2 = np.mean(np.array(losses2))
        writer.add_scalar('BoxLoss/test/', loss1, counter_test)
        writer.add_scalar('LabelLoss/test/', loss2, counter_test)

############################################################################

def getHeatMap(img, box):
    kernel = np.array([[0, 0, 1, 0, 0],[0, 1, 1, 1, 0],[1, 1, 1, 1, 1],[0, 1, 1, 1, 0],[0, 0, 1, 0, 0]]).astype(np.uint8)
    heat = torch.zeros(img.shape[:2])
    sy, sx, ey, ex = box
    sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)
    heat[sx:ex, sy:ey] += 1
    heat = cv2.dilate(np.float32(heat), kernel= kernel, iterations=1)
    heat = cv2.GaussianBlur(np.array(heat) ,(5, 5), sigmaX=0)
    return heat
############################################################################

# Calculates the degree of intersection for two bounding boxes
def intersection_over_union(gt_box, pred_box):
    combined = torch.stack((gt_box, pred_box), dim=1)
    max_0 = torch.max(combined[:, :, 0].T, dim = 0).values
    max_1 = torch.max(combined[:, :, 1].T, dim=0).values
    stacked = torch.stack((gt_box[:, 0] + gt_box[:, 2], pred_box[:, 0] + pred_box[:, 2]), dim=0)
    min_0 = torch.min(stacked, dim = 0).values
    stacked = torch.stack((gt_box[:, 1] + gt_box[:, 3], pred_box[:, 1] + pred_box[:, 3]), dim=0)
    min_1 = torch.min(stacked, dim = 0).values
    w = min_0 - max_0
    h = min_1 - max_1
    intersection = w*h
    union = gt_box[:,2] * gt_box[:,3] + pred_box[:,2] * pred_box[:,3] - intersection
    iou = intersection / union
    binaryIOU = iou.ge(0.5).int()
    return iou, intersection, union, binaryIOU

############################################################################

# Calculates the mean average precision
def meanAP(gt_box, pred_heatMap, labelsPred, labelsTrue):
    pred_box = getBoxFromHeatMap(pred_heatMap)
    softmax = nn.Softmax()
    labelsPred = softmax(labelsPred)
    confidenceCorrectLabel = torch.tensor([labelsPred[i][labelsTrue[i]]  for i in range(0, len(labelsTrue))])
    iou, intersection, union, binaryIOU = intersection_over_union(gt_box, pred_box)
    limits = np.arange(start=0.0, stop=1.0, step=0.05)
    precicions=[]
    for limit in limits:
        corrects = 0
        for j in range(0, len(labelsTrue)):
            if confidenceCorrectLabel[j] >=limit and (iou[j]<=0.5 or labelsTrue[j]==0):
                corrects+=1
        precicion = corrects/len(labelsTrue)
        precicions.append(precicion)
    mAP = np.mean(np.array(precicions))
    return mAP

############################################################################

def getBoxFromHeatMap(heatmap):
    boxes = []
    for heat in heatmap:
        gray = heat[0].detach().numpy().astype(np.uint8)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = np.array(cnts)
        if len(cnts)>0:
            cnts = cnts.reshape((cnts.shape[1], 2))
            sx = cnts[0][0]
            sy = cnts[0][1]
            ex = cnts[-1][0]
            ey = cnts[2][1]
            boxes.append([sx, sy, ex, ey])
            plt.plot([sx, sx], [sy, ey], lw=4, c="blue", label="Predicted")
            plt.plot([sx, ex], [sy, sy], lw=4, c="blue")
            plt.plot([ex, ex], [ey, sy], lw=4, c="blue")
            plt.plot([ex, sx], [ey, ey], lw=4, c="blue")
        else:
            boxes.append([0,0, 1, 1])
    return torch.tensor(boxes)
