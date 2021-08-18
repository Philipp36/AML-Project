import sys

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

############################################################################
from torch import nn
from tqdm import trange


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

def trainLoop(model, optimizer, epochs, train_dataloader, test_dataloader, counter_test, writer, pathModel):

    testLoop(model, test_dataloader, counter_test, writer)
    counter_test += 1
    lossBoxes = nn.MSELoss()
    lossLabels = nn.CrossEntropyLoss()
    for epoch in range(0, epochs):
        print(" Epoch:  ", epoch)
        iterator = iter(train_dataloader)
        losses1 = []
        losses2 = []
        for index in trange(0, len(train_dataloader)):
            image, heat_resized, truth, box = iterator.next()
            optimizer.zero_grad()
            heat_pred, label_pred = model(image)
            loss1 = lossBoxes(heat_pred.double(), box.double())
            loss2 = lossLabels(label_pred, truth)
            if epoch == 0 and index == 0:
                W = float((loss1 / loss2))
            LOSS = loss1 + (W * loss2)
            LOSS.backward()
            optimizer.step()
            losses1.append(float(loss1))
            losses2.append(float(loss2))

        loss1 = np.mean(np.array(losses1))
        loss2 = np.mean(np.array(losses2))
        writer.add_scalar('BoxLoss/train/', loss1, epoch)
        writer.add_scalar('LabelLoss/train/', loss2, epoch)
        testLoop(model, test_dataloader, counter_test, writer)
        counter_test += 1
        torch.save(model, pathModel)

############################################################################

def testLoop(model, test_dataloader, counter_test, writer):
    print("Test ", counter_test)
    lossBoxes = nn.MSELoss()
    lossLabels = nn.CrossEntropyLoss()
    iterator = iter(test_dataloader)
    losses1 = []
    losses2 = []
    for index in range(0, len(test_dataloader)):
        image, heat_resized, truth, box = iterator.next()
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
    heat = torch.zeros_like(img)
    sy, sx, ey, ex = box
    sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)
    heat[sx:ex, sy:ey] += 1
    heat = cv2.dilate(np.float32(heat), kernel= kernel, iterations=1)
    heat = cv2.GaussianBlur(np.array(heat) ,(5, 5), sigmaX=0)
    return heat

############################################################################

def getBoxFromHeatMap(heat):
    return None