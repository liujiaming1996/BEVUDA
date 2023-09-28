import torch.nn as nn
import torch
import torch.nn.functional as F

# class Disc_vox_source(nn.Module):

#     def __init__(self):
#         super(Disc_vox_source, self).__init__()
#         # [24, 80, 112, 16, 44]
#         self.conv1 = torch.nn.Conv3d(80, 128, 3)
#         self.conv2 = torch.nn.Conv3d(128, 256, 3)
#         self.conv3 = torch.nn.Conv3d(256, 512, 3)
#         self.pool = torch.nn.AdaptiveAvgPool3d((1,1,1))
#         self.flat = torch.nn.Flatten(1) # ([4, 6*512*16*44])
#         self.linear1 = torch.nn.Linear(512,256)
#         self.bn1 = torch.nn.BatchNorm1d(256)
#         self.relu1 = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(256,128)
#         self.bn2 = torch.nn.BatchNorm1d(128)
#         self.relu2 = torch.nn.ReLU()
#         self.linear3 = torch.nn.Linear(128,2)
#         self.softmax = torch.nn.LogSoftmax(dim=1)

#     def forward(self, input):
#         x = F.relu(self.conv1(input))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = self.flat(x)
#         x = self.linear1(x)
#         x = self.relu1(self.bn1(x))
#         x = self.linear2(x)
#         x = self.relu2(self.bn2(x))
#         x = self.linear3(x)
#         x = self.softmax(x)
#         return x


# class Disc_vox_target(nn.Module):
    
#     def __init__(self):
#         super(Disc_vox_target, self).__init__()
#         self.conv1 = torch.nn.Conv3d(80, 128, 3)
#         self.conv2 = torch.nn.Conv3d(128, 256, 3)
#         self.conv3 = torch.nn.Conv3d(256, 512, 3)
#         self.pool = torch.nn.AdaptiveAvgPool3d((1,1,1))
#         self.flat = torch.nn.Flatten(1) # ([4, 6*512*16*44])
#         self.linear1 = torch.nn.Linear(512,256)
#         self.bn1 = torch.nn.BatchNorm1d(256)
#         self.relu1 = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(256,128)
#         self.bn2 = torch.nn.BatchNorm1d(128)
#         self.relu2 = torch.nn.ReLU()
#         self.linear3 = torch.nn.Linear(128,2)
#         self.softmax = torch.nn.LogSoftmax(dim=1)

#     def forward(self, input):
#         x = F.relu(self.conv1(input))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = self.flat(x)
#         x = self.linear1(x)
#         x = self.relu1(self.bn1(x))
#         x = self.linear2(x)
#         x = self.relu2(self.bn2(x))
#         x = self.linear3(x)
#         x = self.softmax(x)
#         return x

class Disc_vox_source(nn.Module):

    def __init__(self):
        super(Disc_vox_source, self).__init__()
        # [24, 80, 112, 16, 44]
        self.conv1 = torch.nn.Conv3d(80, 80, 3)
        self.conv2 = torch.nn.Conv3d(80, 80, 3)
        self.conv3 = torch.nn.Conv3d(80, 80, 3)
        self.pool = torch.nn.AdaptiveAvgPool3d((1,1,1))
        self.bn1 = torch.nn.BatchNorm3d(80)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool3d(2)
        self.bn2 = torch.nn.BatchNorm3d(80)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool3d(2)
        self.bn3 = torch.nn.BatchNorm3d(80)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool3d(2)

        self.out = torch.nn.Linear(80,1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = self.pool2(x)

        x = self.pool(x)
        x = x.view(-1,80)
        x = self.out(x)

        return torch.sigmoid(x)


class Disc_vox_target(nn.Module):
    
    def __init__(self):
        super(Disc_vox_target, self).__init__()
        self.conv1 = torch.nn.Conv3d(80, 80, 3)
        self.conv2 = torch.nn.Conv3d(80, 80, 3)
        self.conv3 = torch.nn.Conv3d(80, 80, 3)
        self.pool = torch.nn.AdaptiveAvgPool3d((1,1,1))
        self.bn1 = torch.nn.BatchNorm3d(80)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool3d(2)
        self.bn2 = torch.nn.BatchNorm3d(80)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool3d(2)
        self.bn3 = torch.nn.BatchNorm3d(80)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool3d(2)

        self.out = torch.nn.Linear(80,1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = self.pool2(x)

        x = self.pool(x)
        x = x.view(-1,80)
        x = self.out(x)

        return torch.sigmoid(x)