# DenseNet169 baseline

We provide here the PyTorch training script to train and test the model in this repo.

## Requirements

The main requirements are listed below:

* Pytorch
* re
* skimage
* torchvision
* Python 3.7
* Numpy
* OpenCV
* Scikit-Learn

<!---
# Dataset Split
See Data-split. Patient distribution in each set will be updated soon.
--->
## Environment
The code is based on Python 3.7 and PyTorch 1.3.
The code is run and tested on one GTX1080Ti


# Steps to generate the dataset used to do Training and Testing
1. Download images from the repo `Images-processed`
2. Download .txt files for image names in train, val, and test set from `Data-split` repo
3. Use the dataloader defined in line `80` of the script `DenseNet_predict.py` and load the dataset


# Dataset Distribution
<!---
--->
Images distribution
|  Type | NonCOVID-19 | COVID-19 |  Total |
|:-----:|:-----------:|:--------:|:------:|
| train |      234    |    191   |   425  |
|  val  |       58    |     60   |   118  |
|  test |      105    |     98   |   203  |

Patients distribution
|  Type |    NonCOVID-19   | COVID-19 |  Total |
|:-----:|:----------------:|:--------:|:------:|
| train |        105       |  1-130   |   235  |
|  val  |         24       | 131-162  |    56  |
|  test |         42       | 163-216  |    96   |



* Max CT scans per patient in COVID: 16.0 (patient 2)
* Average CT scans per patient in COVID: 1.6
* Min CT scans per patient in  COVID: 1.0
<!---
Patients frequency ('ID:number')
* train: 12:18  13:9  14:2  15:12  17:20  18:16  19:12  21:8  23:40  24:22  25:11  34:12
* val: 6:26  16:10  27:22 
* test: 7:4  8:8  10:8  11:3  20:12
--->


## Training and Evaluation
   In [144] of the script.
   
   Training is defined in line `190` of the script and validation is defined in line `241`. 
   In line `488`, start the loading of DenseNet-169 model and do training in line `535`.
   You can either train by transfering from the ImageNet pretrained model or train from scratch by setting `pretrain = false`.  
   
   The performance on val set is observed in line `561`. It will print the target value list and the predict value list per epoch. The F1-score, accuray, and AUC of 10 models are printed as a major vote result per 10 epoch. 

## Test
   In [145] of the script. Line `617`. 

## Initial result
   F1:  0.854
   
   ACC: 0.847
   
   AUC: 0.919
