# Deep Learning-Based Semantic Segmentation of Microscale Objects

This is the source code for the deep learning model proposed in the paper [Deep Learning-Based Semantic Segmentation of Microscale Objects](https://arxiv.org/abs/1907.03576). A condensed version of the paper is published in the Proceedings of the [2019 International Conference on Manipulation, Automation, and Robotics at Small Scales](https://marss-conference.org/).


## Data Annotation

Low contrast, bright-field images of multiple human endothelial cells and silica beads shown in the [paper](https://arxiv.org/abs/1907.03576) are considered. Segmentation labels for the images are created using [LabelMe](https://github.com/wkentaro/labelme).

## Implementation

This code is tested on a workstation running Windows 10 operating system, equipped with a 3.7GHz 8 Core Intel Xeon W-2145 CPU, GPU ZOTAC GeForce GTX 1080 Ti, and 64 GB RAM.

Implemented with:
* Tensorflow-gpu 1.10.1
* Keras 2.2.4
* OpenCV 3.4.2
* Python 3.5.5 

## Code
1. Create a 'data' folder in the cell-segmentation-master directory. Create sub-folders 'src' and 'label' in the data folder and place all the dataset images and the corresponding labels in the src and label folders respectively.
2. Run gen_img_aug.py to split the data into training and test sets.
3. Run pre_processing.py to perform pre-processing on the training set.
4. Run train_unet_xception_resnetblock.py to train the model. The model is saved when the validation loss reaches its minimum and is named as unet_xception_resnet_nsgd32_lovaszsoftmax_best.h5.
5. Run predict_lovasz_loss.py to generate segmentation masks for the test images using the saved model.
6. Run get_contour.py to detect the contours of the segmented regions and overlay them on the original image.

## Acknowledgement
* The code for the model is built on the code by [Siddharta](https://github.com/sidml/Image-Segmentation-Challenge-Kaggle) proposed in the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge) hosted by Kaggle.
* Implementation of the Lovasz Softmax loss is as provided by [Maxim Berman](https://github.com/bermanmaxim/LovaszSoftmax).

## Citation
Please cite the following paper if the code is found useful.

```bash
@article{samani2019deep,
  title={Deep Learning-Based Semantic Segmentation of Microscale Objects},
  author={Samani, Ekta U and Guo, Wei and Banerjee, Ashis G},
  journal={arXiv preprint arXiv:1907.03576},
  year={2019}
}
```
