#!/usr/bin/env python3

# Fix for PyInstaller
from os import environ
environ["PYTORCH_JIT"] = "0"

# Import stuff we will need
import torch
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms
import torchvision.models as models
import glob
from os import path
from sys import argv, exit

# Constants

# Initialize torchvision transformer
normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])
val_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

# Function Definitions

def image_load(path):
        ret = Image.open(path).convert('RGB')
        ret = val_transformer(ret).float()
        ret = ret.clone().detach()
        ret = ret.unsqueeze(0)
        if torch.cuda.is_available():
            ret = ret.cuda()
        return ret

# Code for run as main

if __name__ == "__main__":
    # Check to ensure correct argument usage
    if len(argv) != 2:
        print("Usage: ./predict.py image.jpg")
        exit(1)

    # Load image into memory
    image = image_load(argv[1])

    # Get model location
    selftrans = path.dirname(path.realpath(__file__)) + "/Self-Trans.pt"

    # Load model
    model = models.densenet169(pretrained=True)
    if torch.cuda.is_available():
        model = model.cuda()
    pt_net = torch.load(selftrans, map_location=torch.device('cpu'))
    model.load_state_dict(pt_net)
    model.eval()

    # Run model on provided image
    output = model(image)

    # Calculate and print result
    pred = int(output.argmax().item())
    if pred == 0:
        print("This patient is predicted to be COVID-19 positive.")
    elif pred == 1:
        print("This patient is predicted to be COVID-19 negative.")
    else:
        print("An unknown error has occurred and the model cannot evaluate the provided data.")
