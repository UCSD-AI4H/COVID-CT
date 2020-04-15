import os
import cv2
import numpy


DATADIR = "CT_Scan"
CATEGORIES = ["CT_COVID","CT_NonCOVID"]


for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        numpy.savetxt(category + ".csv", img_array, delimiter=',')
