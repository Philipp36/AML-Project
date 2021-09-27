import math
import random
import sys
from math import sin, cos

import cv2
import numpy as np
import torch
from torch import Tensor
from PIL import Image
from matplotlib import pyplot as plt, cm
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from collections.abc import Sequence
import torchvision
############################################################################
from torch import nn, optim, profiler
import torchvision.transforms.functional as F
from tqdm import trange
import cv2


def rescale_box(box, img_size_orig, img_size_new):
    """
        rescales box for visualization and inference
    """
    orig_w, orig_h = img_size_orig
    new_w, new_h = img_size_new
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    sx, sy, ex, ey = box
    return [sx * scale_x, sy * scale_y, ex * scale_x, ey * scale_y]


def rescaleBox(box_pred, box_true, img, imageOrig, transformPredicted = False):
    """
        rescales box for visualization and inference
    """
    # Transform the predicted Box
    if transformPredicted:
        sx, sy, ex, ey = box_pred
        x, y, z = img.shape
        x1, y1, z1 = imageOrig.shape
        scaleX = x1/x
        scaleY = y1/y

    # Transform the true Box
    else:
        sx, sy, ex, ey = box_true
        x, y, z = img.shape
        x1, y1, z1 = imageOrig.shape
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
        ## Draw resized
        #BOX_TRUTH = rescaleBox(box_pred, box_true, IMG, IMG_ORIG, transformPredicted=False)

        sx, sy, ex, ey = box_true

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
    # logging and testing parameters
    logging_interval = 30
    test_interval = 75

    testLoop(model, test_dataloader, counter_test, writer, dev=dev)
    counter_test += 1

    lossBoxes = IoULoss()
    lossLabels = nn.CrossEntropyLoss()
    model.train()
    counter_train = 1

    losses1 = []
    losses2 = []
    #with torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=2),
    #                            on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/train/amermodel'),
    #                            record_shapes=True,
    #                            with_stack=True,
    #                            profile_memory=True) as prof:
    for epoch in range(0, epochs):
        print(" Epoch:  ", epoch)
        iterator = iter(train_dataloader)
        for index in trange(0, len(train_dataloader)):
            image, truth, heat = iterator.next()
            image, truth, heat = image.to(dev), truth.to(dev), heat.to(dev)
            optimizer.zero_grad()
            heat_pred, label_pred = model(image)

            for k in range(0, len(heat_pred)):
                if truth[k] == 0:
                    heat_pred[k] = heat[k]

            loss1 = lossBoxes(heat_pred, heat)
            loss2 = lossLabels(label_pred, truth)
            LOSS = loss1 + loss2
            LOSS.backward()
            optimizer.step()
            # prof.step()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            # LOGGING STEP LOSSES
            if index % logging_interval == 0 and index > 2:
                loss1 = np.mean(np.array(losses1[-logging_interval:]))
                loss2 = np.mean(np.array(losses2[-logging_interval:]))
                writer.add_scalar('BoxLoss/train_step', loss1, counter_train)
                writer.add_scalar('LabelLoss/train_step', loss2, counter_train)
            counter_train += 1
                # losses1 = []
                # losses2 = []

            del LOSS, loss1, loss2, image, truth, heat
            # TESTING AND MODEL SAVING
            if index % test_interval == 0 and index > 2:
                testLoop(model, test_dataloader, counter_test, writer, dev=dev)
                torch.save(model, pathModel)
                counter_test += 1

        # LOGGING EPOCH LOSSES
        loss1 = np.mean(np.array(losses1[-len(train_dataloader):]))
        loss2 = np.mean(np.array(losses2[-len(train_dataloader):]))
        writer.add_scalar('BoxLoss/train_epoch', loss1, epoch)
        writer.add_scalar('LabelLoss/train_epoch', loss2, epoch)


############################################################################

def testLoop(model, test_dataloader, counter_test, writer, dev=torch.device('cpu')):
    with torch.no_grad(): # and \
        # torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=2),
        #                       on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/test/amermodel'),
        #                       record_shapes=False,
        #                       with_stack=False,
        #                       profile_memory=True) as prof:
        model.eval()
        print("Test ", counter_test)
        lossBoxes = IoULoss()
        lossLabels = nn.CrossEntropyLoss()
        iterator = iter(test_dataloader)
        losses1 = []
        losses2 = []
        mean_aps = []
        correct = 0
        for index in trange(0, len(test_dataloader)):
            image, truth, heat = iterator.next()
            image, truth, heat = image.to(dev), truth.to(dev), heat.to(dev)
            heat_pred, label_pred = model(image)

            # heat_pred = torch.where(torch.unsqueeze(truth, 1).expand(-1, 4) == 0, box, heat_pred)
            for k in range(0, len(heat_pred)):
                if truth[k] == 0:
                    heat_pred[k] = heat[k]

            loss1 = lossBoxes(heat_pred, heat)
            loss2 = lossLabels(label_pred, truth)
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            correct += torch.sum(torch.argmax(label_pred, dim=1) == truth)
            #mean_aps.append(meanAP(box, heat_pred, label_pred, truth))

            # del loss1, loss2, image, heat_resized, truth, box


        visualize(image, idx_pred=label_pred, idx_truth=truth, heat_pred=heat_pred, boxes_truth=getBoxFromHeatMap(heat),
                  global_step=counter_test, writer=writer)
        loss1 = float(np.mean(np.array(losses1)))
        loss2 = float(np.mean(np.array(losses2)))
        #mean_ap = float(np.mean(np.array(mean_aps)))
        writer.add_scalar('BoxLoss/test/', loss1, counter_test)
        writer.add_scalar('LabelLoss/test/', loss2, counter_test)
        #writer.add_scalar('mean_ap/test/', mean_ap, counter_test)
        iterator = iter(test_dataloader)
        batch_size = iterator.next()[0].size()[0]
        writer.add_scalar('val_acc/test', correct / (len(test_dataloader) * batch_size), counter_test)


        model.train()

############################################################################

def getHeatMap(size_w, size_h, boxes):
    kernel = np.array([[0, 0, 1, 0, 0],[0, 1, 1, 1, 0],[1, 1, 1, 1, 1],[0, 1, 1, 1, 0],[0, 0, 1, 0, 0]]).astype(np.uint8)
    heat = torch.zeros(size=(size_h, size_w))
    for box in boxes:
        sx, sy, ex, ey = box
        # TODO: look up direction of tensor in y-axis
        sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)
        heat[sy:ey, sx:ex] += 1
    heat = cv2.dilate(np.float32(heat), kernel= kernel, iterations=1)
    heat = cv2.GaussianBlur(np.array(heat), (5, 5), sigmaX=0)
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


def meanAP(gt_box, pred_box, labelsPred, labelsTrue):
    """
        function used to calculate the mean average precision metric
    """
    # pred_box = getBoxFromHeatMap(pred_heatMap)
    softmax = nn.Softmax(dim=1)
    labelsPred = softmax(labelsPred)
    confidenceCorrectLabel = torch.tensor([labelsPred[i][labelsTrue[i]]  for i in range(0, len(labelsTrue))])
    iou, intersection, union, binaryIOU = intersection_over_union(gt_box, pred_box)
    limits = np.arange(start=0.0, stop=1.0, step=0.05)
    precicions=[]
    for limit in limits:
        corrects = 0
        for j in range(0, len(labelsTrue)):
            if confidenceCorrectLabel[j] >= limit and (iou[j] >= 0.5 or labelsTrue[j] == 0):
                corrects += 1
        precicion = corrects/len(labelsTrue)
        precicions.append(precicion)
    mAP = np.mean(np.array(precicions))
    return mAP


def getBoxFromHeatMap(heatmap):
    """
        function used to convert heatmaps to bounding boxes (batch wise)
        Args:
            heatmap: Tensor containing heatmaps in format [N, H, W]
        Returns:
            boxes_batch: list containing list of all boxes found in corresponding heatmap
    """
    boxes_batch = []
    heatmap = heatmap.detach().cpu().numpy().astype(np.uint8)
    for gray in heatmap:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])
        if len(cnts) == 0:
            boxes.append([0, 0, 1, 1])
        boxes_batch.append(boxes)
    return boxes_batch
    #
    #     try:
    #         if len(cnts)>0:
    #             cnts = cnts.reshape((cnts.shape[1], 2))
    #             sx = cnts[0][0]
    #             sy = cnts[0][1]
    #             ex = cnts[-1][0]
    #             ey = cnts[2][1]
    #             boxes.append([sx, sy, ex, ey])
    #         else:
    #             boxes.append()
    #     except:
    #         return torch.tensor([[0,0, 1, 1]])
    # return torch.tensor(boxes)

############################################################################

def addAugmentation(trainPaths, boundingBoxes, truths):

    # 0: Original image
    # 1: Mirrored image
    # 2: Cropped image
    # 3: Left rotation
    # 4: Right rotation

    trainPathsNew = []
    boundingBoxesNew = []
    truthsNew = []
    types = []
    for i in range(0, len(trainPaths)):

        path = trainPaths[i]
        box = boundingBoxes[i]
        label = truths[i]

        #0. The original image
        trainPathsNew.append(path)
        boundingBoxesNew.append(box)
        truthsNew.append(label)
        types.append(0)

        #1. The Mirrored image
        trainPathsNew.append(path)
        boundingBoxesNew.append(box)
        truthsNew.append(label)
        types.append(1)

        #2. Left rotation
        trainPathsNew.append(path)
        boundingBoxesNew.append(box)
        truthsNew.append(label)
        types.append(2)

        #3. Right rotation
        trainPathsNew.append(path)
        boundingBoxesNew.append(box)
        truthsNew.append(label)
        types.append(3)

    return trainPathsNew, boundingBoxesNew, truthsNew, types

############################################################################

def mirrorImage(img, box):
    sx, sy, ex, ey = box
    X, Y = np.array(img).shape[:2]
    exNew = Y - sx
    sxNew = exNew - (ex - sx)
    boxNew = [sxNew, sy, exNew, ey]
    imgNew = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    return imgNew, boxNew

############################################################################

def cropImage(img, box):
    sx, sy, ex, ey = box
    X, Y = np.array(img).shape[:2]
    resizeFactor = 0.95
    newX = X * resizeFactor
    newY = Y * resizeFactor
    sxNew, syNew, exNew, eyNew = sx * resizeFactor, sy * resizeFactor, ex * resizeFactor, ey * resizeFactor
    boxNew = [sxNew, syNew, exNew, eyNew]
    img_cropped = img.crop((X - newX, Y - newY, newX, newY))
    return img_cropped, boxNew

############################################################################

def rotateImage(img, box, option="left"):
    angle = random.randint(1, 45)
    if option == "right":
        angle = -angle
    img = np.array(img)
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    shiftedIMG = cv2.warpAffine(img, M, (w, h))
    return shiftedIMG, box


def visualize(imgs, idx_pred, idx_truth, heat_pred, boxes_truth, writer, global_step):
    """
        visualization used to display inference in Tensorboard
    """
    idx_to_label = {
        0: "negative",
        1: "typical",
        2: "indeterminate",
        3: "atypical"
    }
    idx_pred = torch.argmax(idx_pred, dim=1)
    label_pred = []
    for idx in idx_pred:
        label_pred.append(idx_to_label[idx.item()])

    label_truth = []
    for idx in idx_truth:
        label_truth.append(idx_to_label[idx.item()])
    images = []
    for i in range(len(imgs)):
        # TRUTH BOUNDING BOX
        img = imgs[i].cpu()
        img = torchvision.utils.draw_bounding_boxes(img, torch.tensor(boxes_truth[i]),
                                                    colors=['red']*len(boxes_truth[i]), fill=False, width=3,
                                                    font_size=36)

        # PREDICTION
        #img = torchvision.utils.draw_bounding_boxes(imgs[i].cpu(), torch.unsqueeze(boxes_pred[i].cpu(), 0),
        #                                                label_pred[i], colors=['blue'], fill=False, width=3,
        #                                                font_size=36)
        # DRAW HEATMAP
        hp = torch.unsqueeze(heat_pred[i], 0).detach().cpu()
        hp_size = list(hp.size())
        hp = torch.cat((torch.zeros(size=(3 - hp_size[0], *hp_size[1:])), hp))
        img = F.to_pil_image(img)
        hp = F.to_pil_image(hp)
        img = Image.blend(img, hp, alpha=0.2)

        # DRAW LABELS
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('OpenSans-Bold.ttf', size=20)
        draw.text((10, 0), label_truth[i], font=font, fill=(255, 0, 0, 255))
        draw.text((10, 30), label_pred[i], font=font, fill=(0, 0, 255, 255))

        img = F.to_tensor(img)
        images.append(torch.unsqueeze(img, dim=0))
        writer.add_images(tag='test/vis', img_tensor=torch.cat(images, dim=0), global_step=global_step, dataformats='NCHW')


def rotate_bb(box, angle, rotc_x, rotc_y):#
    """
     Args:
         box: [xmin, ymin, xmax, ymax] describing a bounding box.
         angle: angle by which the box is to be rotated
         rotc_x: x-coordinate of the rotation center
         rotc_y: y-coordinate of the rotation center
     Returns:
         rotated_box: [xmin, ymin, xmax, ymax]
     """
    xmin, ymin, xmax, ymax = box
    coordinates = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    coordinates = [rotate_coordinate(x, y, angle, rotc_x, rotc_y) for x, y in coordinates]
    return get_bb_from_points(coordinates)


def rotate_coordinate(x, y, angle, rotc_x, rotc_y):
    """
    Helper function to rotate a single coordinate
    Args:
        x: x-coordinate
        y: y-coordinate
        angle: angle by which the coordinate is to be rotated
        rotc_x: x-coordinate of the rotation center
        rotc_y: y-coordinate of the rotation center
    Returns:
        x, y: rotated coordinate
    """
    rad = math.radians(angle)
    return rotc_x + (x - rotc_x) * math.cos(rad) + (y - rotc_y) * math.sin(rad),\
           rotc_y - (x - rotc_x) * math.sin(rad) + (y - rotc_y) * math.cos(rad)


def get_bb_from_points(coordinates):
    """
        Helper function to get bounding box from four points
        Args:
           coordinates: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        Returns:
            box: bb in [xmin,ymin,xmax,ymax] format
        """
    x_min = x_max = coordinates[0][0]
    y_min = y_max = coordinates[0][1]
    for x, y in coordinates[1:]:
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    return x_min, y_min, x_max, y_max
