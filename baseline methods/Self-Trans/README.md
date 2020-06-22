# Self-Trans 


**We provide our current best model `Self-Trans` in this repo**
 
### Environment
The code is based on Python 3.7 and PyTorch 1.3

The MoCO code is run on four GTX1080Ti with batch size 128. The pretrained model is finetuned on one GTX1080Ti.


### Dataset
Use the split in `Data-split`. 
To generate the dataset for training and testing, 
1. Download images from repo `Images-processed`
2. Download txt files for image names in train, val, and test set from `Data-split` repo
3. Use the dataloader defined in line [5] of the script `CT_predict-efficient-pretrain.ipynb` and load the dataset

The input image to the model will be (N, 224, 224, 3), for DenseNet this will be taken as the input, for some other models which require the input to have only 1 channel, you need to change the input data by setting 

       data = data[:, 0, :, :]
       data = data[:, None, :, :] 
       
in [8][9][10] for train, val, and test in the fine-tune script [CT-predict-pretrain.ipynb](CT-predict-pretrain.ipynb) (for our CT images, each channel is the same)

### How to train
We provide you with the PyTorch MoCo pretraining script [main.py](main.py) and fine-tune script [CT-predict-pretrain.ipynb](CT-predict-pretrain.ipynb)

The `Self-Trans` model is trained by two steps:

*First step*: Load the model pretrained on ImageNet. Install the model you want to use, e.g. to use effientNet, call `pip install --upgrade efficientnet-pytorch` to install.  Then locate the [LUNA](LUNA) dataset, change you path in line 48 and 238 of [main_coco.py](main_coco.py), and call `ipython main.py` to run MoCo on `LUNA` dataset. Then run MoCo on `COVID-CT` by change the dataset to `COVID-CT`. To do MoCo, 4 or 8 GPUs are needed. If you use 8 GPUs to train, you need to adjust the preset batch size to 256 (otherwise for 4 GPUs use the default 128).

*Second step*: Load MoCo pretrained model in line [17] of `CT_predict-efficient-pretrain.ipynb` and do finetuning.

### Results
F1: 0.85

Accuracy: 0.86

AUC: 0.94


<p align="center">
	<img src="../gradcam.png" alt="photo not available" width="70%" height="70%">
	<br>
	<em>    Grad-CAM visualizations for DenseNet-169. From left to right: Column (1) are original images with COVID-19; Column (2-3) are Grad-CAM visualizations for the model trained with random initialization; Column (4-5) are Grad-CAM visualizations for ImageNet pretrained model; Column (6-7) are Grad-CAM visualizations for Self-Trans model.</em>
</p>

### Pretrained model
See `Self-Trans.pt` with DenseNet-169 backbone.


### How to use our Pretrained model
We provide an example notebook file `CT_predict-efficient-pretrain.ipynb`, the pretrained model is loaded in [30] . Change the name and path to our provided Self-Trans.pt to load it correctly. The model achieves an F1-score of 0.85 on the test set.


### Reference 
The details of this approach are described in this [preprint](https://www.medrxiv.org/content/10.1101/2020.04.13.20063941v1).

If you find the code useful, please cite:


    @Article{he2020sample,
      author  = {He, Xuehai and Yang, Xingyi and Zhang, Shanghang, and Zhao, Jinyu and Zhang, Yichen and Xing, Eric, and Xie, Pengtao},
      title   = {Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans},
      journal = {medrxiv},
      year    = {2020},
    }

The code of the MoCo part can refer to [https://github.com/facebookresearch/moco](https://github.com/facebookresearch/moco)

 


