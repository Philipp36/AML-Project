import torch
from torch import nn
from torchvision import models


class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        self.maxPool = nn.MaxPool2d(3)
        self.relu = nn.ReLU()

        #Box
        self.fc_out_box = nn.Linear(256, 4)
        #Label
        self.fc_out_labels = nn.Linear(256, 4)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.maxPool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.maxPool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.maxPool(x)
        x = self.relu(x)

        x = x.flatten(start_dim=1)

        box = self.fc_out_box(x)
        labels = self.fc_out_labels(x)
        return box, labels


class AmerBackbone(nn.Module):

    def __init__(self):
        super(AmerBackbone, self).__init__()

        backbone = models.resnet50(pretrained=True, progress=True)
        out_channels = backbone.fc.in_features
        self.layers = list(backbone.children())[:-1]  # + [nn.AvgPool2d(kernel_size=(7, 7), stride=1, padding=0)]

    def forward(self, x):
        m1 = nn.Sequential(*self.layers[:-4])
        x = m1(x)
        x = act56 = self.layers[-4](x)
        x = act28 = self.layers[-3](x)
        x = act14 = self.layers[-2](x)
        x = self.layers[-1](x)
        return (act56, act28, act14), x


class AmerLocHead(nn.Module):

    def __init__(self):
        super(AmerLocHead, self).__init__()

        self.loc1 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=(2, 2)),
                                  nn.Conv2d(kernel_size=(1, 1), stride=1, in_channels=2048, out_channels=1024))
        self.loc2 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=(2, 2)),
                                  nn.Conv2d(kernel_size=(1, 1), stride=1, in_channels=1024, out_channels=512))
        self.loc3 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=(2, 2)),
                                  nn.Conv2d(kernel_size=(3, 3), stride=2, in_channels=512, out_channels=128),
                                  nn.ReLU(inplace=True))

    def forward(self, x, y, z):
        x = self.loc1(x)

        x = self.loc2(x + y)
        x = self.loc3(x + z)
        x, _ = torch.max(x, dim=1, keepdim=True)
        up = nn.Upsample(size=(448, 448), mode='nearest')
        return up(x).squeeze()


class AmerModel(nn.Module):

    def __init__(self):
        super(AmerModel, self).__init__()

        self.backbone = AmerBackbone()
        self.det_head = nn.Sequential(nn.Flatten(),
                                      nn.Linear(in_features=2048, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.3),
                                      nn.Linear(in_features=1024, out_features=256),
                                      nn.ReLU(),
                                      nn.Linear(in_features=256, out_features=4))
        self.loc_head = AmerLocHead()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.expand(-1, 3, -1, -1)
        (act56, act28, act14), x = self.backbone(x)
        return self.loc_head(act14, act28, act56), self.det_head(x)
