# MUAD: Multiple Uncertainties for Autonomous Driving Dataset

This repository contains some development kits including the scripts for the evaluation (in PyTorch) that we used for our BMVC 2022 paper: 

MUAD: Multiple Uncertainties for Autonomous Driving, a benchmark for multiple uncertainty types and tasks.

[[Paper]](https://arxiv.org/abs/2203.01437) [[Website]](https://muad-dataset.github.io/)  [[Data]](https://github.com/ENSTA-U2IS-AI/torch-uncertainty)

## Download and use MUAD on a headless server with TorchUncertainty

You will find a **PyTorch dataset** for MUAD's training and validation sets in **semantic segmentation** and **depth prediction** with **automated download** in [**TorchUncertainty**](https://github.com/ENSTA-U2IS-AI/torch-uncertainty).

## ICCV UNCV 2023 | MUAD challenge
MUAD challenge is now on board on the Codalab platform for uncertainty estimation in semantic segmentation. This challenge is hosted in conjunction with the [ICCV 2023](https://iccv2023.thecvf.com/) workshop, [Uncertainty Quantification for Computer Vision (UNCV)](https://uncv2023.github.io/). Go and have a try! 🚀 🚀 🚀 [[Challenge link]](https://codalab.lisn.upsaclay.fr/competitions/8007)

## How to download the MUAD dataset?

If you need MUAD Dataset, please Click and Fill in this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSfTyCCPoO-MrVWKrp5hqyy4Bp9wZKh2Ww7_0MRnk-uu4Wf1yA/viewform?usp=sf_link).

We provide you with permanent download links as soon as you finish submitting the form.

**[Note]** <u>We will release all the test sets (with the RGB images and the ground truth maps) after the MUAD challenge on the Codalab. Currently, only a small part of the test sets is released with only the RGB images.</u>


## Semantic segmentation

### Training and Evaluation on MUAD
We provide here a training and evaluation example, as well as a checkpoint based on DeepLab v3 plus. 
Github link: [[DeepLabv3Plus-MUAD-Pytorch]](https://github.com/ENSTA-U2IS/DeepLabV3Plus-MUAD-Pytorch).


### Evaluation metrics
See folder `./evaluation_seg/`. This scoring program is also used in our [Challenge](https://codalab.lisn.upsaclay.fr/competitions/8007). Check `./evaluation_seg/evaluation.py` for details.

We provide the implementations on `mECE`, `mAUROC`, `mAUPR`, `mFPR`, `mIoU`, `mAccuracy`, etc., see `./evaluation_seg/stream_metrics.py` for details.

You have to make some modifications in the codes according to your own needs. For instance, we by default set our output confidence map as type `.pth`, and set only `mIOU` for semantic segmentation performance, etc.

### Prediction format
The predictions we generate for semantic segmentation task follow the following format.

For each image_id in the test set, we predict a __confidence score__ and a __predicted class label__ for each pixel. The prediction results are saved as __dictionary objects__ in __`.pth`__ form. Here is an example to show the components in a .pth prediction result:

```python
import torch
prediction = torch.load('000000_leftImg8bit.pth')
print(prediction.keys()) # should output: dict_keys(['conf', 'pred'])
print(prediction['conf']) # should output: torch.Size([1024, 2048])
print(prediction['pred']) # should output: torch.Size([1024, 2048])
```
The confidence score should be __torch.float16__ and the predicted class labels should be __torch.int64__.

### More information
The indexes of the classes of semantic segmentation are the following (in leftLabel):

| **class names**                       | **ID** |
|----------------------------------------|---------|
| road                                   | 0       |
| sidewalk                               | 1       |
| building                               | 2       |
| wall                                   | 3       |
| fence                                  | 4       |
| pole                                   | 5       |
| traffic light                          | 6       |
| traffic sign                           | 7       |
| vegetation                             | 8       |
| terrain                                | 9       |
| sky                                    | 10      |
| person                                 | 11      |
| rider                                  | 12      |
| car                                    | 13      |
| truck                                  | 14      |
| bus                                    | 15      |
| train                                  | 16      |
| motorcycle                             | 17      |
| bicycle                                | 18      |
| bear deer cow                          | 19      |
| garbage_bag stand_food trash_can       | 20      |

## Monocular depth estimation

### Evaluation metrics
See folder `./evaluation_depth/`. We provide two scripts, `./evaluation_depth/depth_metrics.py` is for depth prediction performance, and `./evaluation_depth/sparsification.py` is for sparsification curves and its corresponding uncertainty estimation performance metrics, more details on this metric can be found in [this paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.html) and the [git repo](https://github.com/mattpoggi/mono-uncertainty).

### More information
The depth groundtruth data is in the form of `.exr` files. The depth in the image is annotated as min/max depth value. To load and transfer the depth in the image to the depth in meters, you can try the following codes:
```python
import cv2
from PIL import Image
import numpy as np
depth_path = 'xxx.exr'
depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
depth = Image.fromarray(depth)
depth = np.asarray(depth, dtype=np.float32)
depth = 400 * (1 - depth) # the depth in meters
```

## Object/Instance detection
TODO

## Citation
If you find this work useful for your research, please consider citing our paper:
```
@inproceedings{franchi22bmvc,
  author    = {Gianni Franchi and 
              Xuanlong Yu and 
              Andrei Bursuc and 
              Angel Tena and 
              Rémi Kazmierczak and 
              Severine Dubuisson and 
              Emanuel Aldea and 
              David Filliat},
  title     = {MUAD: Multiple Uncertainties for Autonomous Driving benchmark for multiple uncertainty types and tasks},
  booktitle = {33rd British Machine Vision Conference, {BMVC} 2022},
  year      = {2022}
}
```

## Copyright
Copyright for MUAD Dataset is owned by Université Paris-Saclay (SATIE Laboratory, Gif-sur-Yvette, FR) and ENSTA Paris (U2IS Laboratory, Palaiseau, FR).
