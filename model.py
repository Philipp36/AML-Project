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

        backbone = models.resnet18(pretrained=True, progress=True)
        out_channels = backbone.fc.in_features
        layers = list(backbone.children())[:-1]  # + [nn.AvgPool2d(kernel_size=(7, 7), stride=1, padding=0)]
        self.core = nn.Sequential(*layers[:-4])
        self.act56 = layers[-4]
        self.act28 = layers[-3]
        self.act14 = layers[-2]
        self.out = layers[-1]

    def forward(self, x):
        x = self.core(x)
        x = act56 = self.act56(x)
        x = act28 = self.act28(x)
        x = act14 = self.act14(x)
        x = self.out(x)
        return (act56, act28, act14), x


class AmerLocHead(nn.Module):

    def __init__(self):
        super(AmerLocHead, self).__init__()

        self.loc1 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=(2, 2)),
                                  nn.Conv2d(kernel_size=(1, 1), stride=1, in_channels=512, out_channels=256))
        self.loc2 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=(2, 2)),
                                  nn.Conv2d(kernel_size=(1, 1), stride=1, in_channels=256, out_channels=128))
        self.loc3 = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=(2, 2)),
                                  nn.Conv2d(kernel_size=(3, 3), stride=2, in_channels=128, out_channels=64),
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
                                      nn.Linear(in_features=512, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.3),
                                      nn.Linear(in_features=256, out_features=128),
                                      nn.ReLU(),
                                      nn.Linear(in_features=128, out_features=4))
        self.loc_head = AmerLocHead()

    def forward(self, x):
        (act56, act28, act14), x = self.backbone(x)
        return self.loc_head(act14, act28, act56), self.det_head(x)
