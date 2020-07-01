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
from PyQt5.QtWidgets import *
from time import sleep

# ---- Constants ----

# Initialize torchvision transformer
normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])
val_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

# ---- Neural Functions ----

def image_load(path):
        ret = Image.open(path).convert('RGB')
        ret = val_transformer(ret).float()
        ret = ret.clone().detach()
        ret = ret.unsqueeze(0)
        if torch.cuda.is_available():
            ret = ret.cuda()
        return ret

# ---- PyQt5 Class ----

class Predict(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("COVID-19 Predictor")
		self.setFixedSize(400, 250)
		self.openFileButton = QPushButton("Open File")
		self.openFileButton.clicked.connect(self.eval)
		self.label = QLabel("Prediction: (not run)")
		self.layout = QVBoxLayout()
		self.layout.addWidget(self.openFileButton)
		self.layout.addWidget(self.label)
		self.setLayout(self.layout)
		self.show()
	def openFileDialog(self):
		ops = QFileDialog.Options()
		ops |= QFileDialog.DontUseNativeDialog
		imageName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files(*)", options=ops)
		return imageName
	def eval(self):
		# Get path to image
		imagePath = self.openFileDialog()
		if not imagePath:
			return
			
		# Show user that it is in progress
		self.label.setText("Prediction: (evaluating)")

		# Load image into memory
		image = image_load(imagePath)
		
		# Load model
		model = models.densenet169(pretrained=True)
		pt_net = torch.load("Self-Trans.pt", map_location=torch.device('cpu'))
		model.load_state_dict(pt_net)
		model.eval()
		
		# Run model on image
		output = model(image)
		
		# Calculate and print result
		pred = int(output.argmax().item())
		if pred == 0:
			self.label.setText("Prediction: COVID-19 positive")
		elif pred == 1:
			self.label.setText("Prediction: COVID-19 negative")
		else:
			self.label.setText("Unknown error while running model.")
			
		
	
# ---- Main ----

if __name__ == '__main__':
	app = QApplication(argv)
	inst = Predict()
	exit(app.exec_())
