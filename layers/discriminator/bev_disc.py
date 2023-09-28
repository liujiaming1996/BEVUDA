import torch.nn as nn
import torch
import torch.nn.functional as F

# class Disc_bev_source(nn.Module):
    
#     def __init__(self):
#         super(Disc_bev_source, self).__init__()
#         # [4, 80, 128, 128]
#         self.conv1 = nn.Conv2d(80, 128, kernel_size=5)
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=5)
#         self.conv3 = nn.Conv2d(256,512, kernel_size=5)
#         # self.conv3_drop = nn.Dropout2d()
#         self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
#         self.flat = torch.nn.Flatten(1) # ([4, 6*512*16*44])
#         self.linear1 = torch.nn.Linear(512,64)
#         self.bn = torch.nn.BatchNorm1d(64)
#         self.relu = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(64,2)
#         self.softmax = torch.nn.LogSoftmax(dim=1)


#     def forward(self, input):

#         x = F.relu(self.conv1(input))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = self.flat(x)
#         x = self.linear1(x)
#         x = self.relu(self.bn(x))
#         x = self.linear2(x)
#         x = self.softmax(x)

#         return x


# class Disc_bev_target(nn.Module):

#     def __init__(self):
#         super(Disc_bev_target, self).__init__()
#         # [4, 80, 128, 128]
#         self.conv1 = nn.Conv2d(80, 128, kernel_size=5)
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=5)
#         self.conv3 = nn.Conv2d(256,512, kernel_size=5)
#         self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
#         self.flat = torch.nn.Flatten(1) # ([4, 6*512*16*44])
#         self.linear1 = torch.nn.Linear(512,64)
#         self.bn = torch.nn.BatchNorm1d(64)
#         self.relu = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(64,2)
#         self.softmax = torch.nn.LogSoftmax(dim=1)

#     def forward(self, input):

#         x = F.relu(self.conv1(input))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = self.flat(x)
#         x = self.linear1(x)
#         x = self.relu(self.bn(x))
#         x = self.linear2(x)
#         x = self.softmax(x)

#         return x


# PatchGan
class Disc_bev_source(nn.Module):
    
    def __init__(self):
        super(Disc_bev_source, self).__init__()
        # [4, 80, 128, 128]
        self.model = nn.Sequential (
                    nn.Conv2d(80, 80, 4, 1, 0),
                    nn.BatchNorm2d(80),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(80, 80, 3, 2, 1),
                    nn.BatchNorm2d(80),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(80, 80, 3, 2, 1),
                    nn.BatchNorm2d(80),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(80, 80, 3, 2, 1),
                    nn.BatchNorm2d(80),
                    nn.ReLU(True)
                )
        self.out = nn.Linear(320, 1)


    def forward(self, x):

        x = self.model(x)
        x = x.view(-1, 320)
        x = self.out(x)
        return torch.sigmoid(x)


class Disc_bev_target(nn.Module):
    
    def __init__(self):
        super(Disc_bev_target, self).__init__()
        # [4, 80, 128, 128]
        self.model = nn.Sequential (
                    nn.Conv2d(80, 80, 4, 1, 0),
                    nn.BatchNorm2d(80),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(80, 80, 3, 2, 1),
                    nn.BatchNorm2d(80),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(80, 80, 3, 2, 1),
                    nn.BatchNorm2d(80),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(80, 80, 3, 2, 1),
                    nn.BatchNorm2d(80),
                    nn.ReLU(True)
                )
        self.out = nn.Linear(320, 1)


    def forward(self, x):

        x = self.model(x)
        x = x.view(-1, 320)
        x = self.out(x)
        return torch.sigmoid(x)
