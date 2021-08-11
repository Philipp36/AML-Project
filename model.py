import torch
from torch import nn


class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        self.maxPool = nn.MaxPool2d(3)
        self.relu = nn.ReLU()

        #Box
        self.fc_out_box = nn.Linear(16, 4)
        #Label
        self.fc_out_labels = nn.Linear(16, 4)

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
