import sys
import torch
import pytorch_lightning as pl
from torch import nn, optim
from helperFunctions import *

############################################################################

class trainerLightning(pl.LightningModule):
    def __init__(self, model, config, dev=torch.device('cpu')):
        super(trainerLightning, self).__init__()
        self.model = model
        self.config = config
        self.lossBoxes = nn.MSELoss()
        self.lossLabels = nn.CrossEntropyLoss()
        self.dev = dev

    def training_step(self, batch, batch_idx):
        image, heat_resized, truth, box = batch
        image, heat_resized, truth, box = image.to(self.dev), heat_resized.to(self.dev), truth.to(self.dev), box.to(self.dev)
        heat_pred, label_pred = self.model(image)
        #mAP = meanAP(box, heat_pred, label_pred, truth)
        #self.log("mAP/Train", mAP)
        loss1 = self.lossBoxes(heat_pred.double(), heat_resized.double())
        loss2 = self.lossLabels(label_pred, truth)
        self.log("HeatMapLoss/Train", loss1, on_step=True, on_epoch=True)    # TODO: switch to mean over epoch trough on_epoch=True?
        self.log("LabelLoss/Train", loss2, on_step=True, on_epoch=True)      # TODO: switch to mean over epoch
        LOSS = loss1 + loss2
        return LOSS

    def validation_step(self, batch, batch_idx):
        image, heat_resized, truth, box = batch
        image, heat_resized, truth, box = image.to(self.dev), heat_resized.to(self.dev), truth.to(self.dev), box.to(self.dev)
        heat_pred, label_pred = self.model(image)
        #mAP = meanAP(box, heat_pred, label_pred, truth)
        #self.log("mAP/Test", mAP)
        loss1 = self.lossBoxes(heat_pred.double(), heat_resized.double())
        loss2 = self.lossLabels(label_pred, truth)
        self.log("HeatMapLoss/Test", loss1, on_step=True, on_epoch=True)     # TODO: log val accuracy (mean average precision) instead of loss
        self.log("LabelLoss/Test", loss2, on_step=True, on_epoch=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), **self.config['optimizer'])
        return optimizer
