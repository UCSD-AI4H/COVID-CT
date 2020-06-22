###############################
## This document created by Alexandre Boulch, ONERA, France is
## distributed under GPL license
###############################

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SegNet(nn.Module):

    #####
    # Weight layers definition
    def __init__(self,label_nbr): #label nbr is the dimension of the output
        super(SegNet, self).__init__()

        # Stage 1: 3 -> 64
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Stage 2: 64 -> 128
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Stage 3: 128 -> 256
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Stage 4: 256 -> 512 -> 256
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        # Stage 5: 256 -> 128
        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # Stage 6: 128 -> 64
        self.conv6_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Stage 7: 64 -> label_nbr
        self.conv7_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv7_3 = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    ######
    ## Network definition
    def forward(self, x):

        x1_1 = F.relu(self.conv1_1(x))
        x1_2 = F.relu(self.conv1_2(x1_1))

        x1_p, id1 = F.max_pool2d(x1_2,kernel_size=2, stride=2,return_indices=True)

        x2_1 = F.relu(self.conv2_1(x1_p))
        x2_2 = F.relu(self.conv2_2(x2_1))

        x2_p, id2 = F.max_pool2d(x2_2,kernel_size=2, stride=2,return_indices=True)

        x3_1 = F.relu(self.conv3_1(x2_p))
        x3_2 = F.relu(self.conv3_2(x3_1))
        x3_3 = F.relu(self.conv3_3(x3_2))

        x3_p, id3 = F.max_pool2d(x3_3,kernel_size=2, stride=2,return_indices=True)

        x4_1 = F.relu(self.conv4_1(x3_p))
        x4_2 = F.relu(self.conv4_2(x4_1))
        x4_3 = F.relu(self.conv4_3(x4_2))

        xd1_1 = F.max_unpool2d(x4_3, id3, kernel_size=2, stride=2)

        x5_1 = F.relu(self.conv5_1(xd1_1))
        x5_2 = F.relu(self.conv5_2(x5_1))
        x5_3 = F.relu(self.conv5_3(x5_2))

        xd2_1 = F.max_unpool2d(x5_3, id2, kernel_size=2, stride=2)

        x6_1 = F.relu(self.conv6_1(xd2_1))
        x6_2 = F.relu(self.conv6_2(x6_1))

        xd3_1 = F.max_unpool2d(x6_2, id1, kernel_size=2, stride=2)

        x7_1 = F.relu(self.conv7_1(xd3_1))
        x7_2 = F.relu(self.conv7_2(x7_1))
        x7_3 = F.relu(self.conv7_3(x7_2))

        return x7_3

    ######
    # load weights
    def load_weights(self, filename):

        corresp_name = {
        "features.0.weight":"conv1_1.weight",
        "features.0.bias":"conv1_1.bias",
        "features.2.weight":"conv1_2.weight",
        "features.2.bias":"conv1_2.bias",
        "features.5.weight":"conv2_1.weight",
        "features.5.bias":"conv2_1.bias",
        "features.7.weight":"conv2_2.weight",
        "features.7.bias":"conv2_2.bias",
        "features.10.weight":"conv3_1.weight",
        "features.10.bias":"conv3_1.bias",
        "features.12.weight":"conv3_2.weight",
        "features.12.bias":"conv3_2.bias",
        "features.14.weight":"conv3_3.weight",
        "features.14.bias":"conv3_3.bias",
        "features.17.weight":"conv4_1.weight",
        "features.17.bias":"conv4_1.bias",
        "features.19.weight":"conv4_2.weight",
        "features.19.bias":"conv4_2.bias",
        }
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(filename) # load the weigths
        for name in th:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(s_dict)
