import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url


class CNN(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Linear(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        super(Linear, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(224*224*3, num_classes),
        )

    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MLP(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(224*224*3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64,num_classes)
        )

    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self,num_classes=1000,):
        super(SimpleCNN, self).__init__()  # b, 3, 32, 32
        layer1 = torch.nn.Sequential()
        layer1.add_module('conv1', torch.nn.Conv2d(3, 32, 3, 1, padding=1))

        layer1.add_module('relu1', torch.nn.ReLU(True))
        layer1.add_module('pool1', torch.nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer3 = torch.nn.Sequential()
        layer3.add_module('fc1', torch.nn.Linear(401408, num_classes))
        self.layer3 = layer3

    def forward(self, x):
        conv1 = self.layer1(x)
        fc_input = conv1.view(conv1.size(0), -1)
        fc_out = self.layer3(fc_input)
        return fc_out