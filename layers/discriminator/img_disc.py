import torch.nn as nn
import torch

class Disc_img_source(nn.Module):

    def __init__(self):
        super(Disc_img_source, self).__init__()
        # [4, 6, 512, 16, 44]
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flat = torch.nn.Flatten(1)
        self.linear1 = torch.nn.Linear(3072,256)
        self.bn = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256,2)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.adv_loss = torch.nn.NLLLoss()

    def forward(self, input):
        x = self.pool(input)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.relu(self.bn(x))
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class Disc_img_target(nn.Module):
    
    def __init__(self):
        super(Disc_img_target, self).__init__()

        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flat = torch.nn.Flatten(1) # ([4, 6*512*16*44])
        self.linear1 = torch.nn.Linear(3072,256)
        self.bn = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256,2)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.adv_loss = torch.nn.NLLLoss()

    def forward(self, input):
        x = self.pool(input)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.relu(self.bn(x))
        x = self.linear2(x)
        x = self.softmax(x)
        return x
            