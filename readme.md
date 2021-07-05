# Problem Statement

Detect and localize surface defects found on a steel sheets. There are total 4 types of defects on a steel sheet in the given dataset.

The dataset contains steel sheet images, with type of defect the image has and location of defect on the steel surface

Detail information about problem statement and dataset can be found [here](https://www.kaggle.com/c/severstal-steel-defect-detection)

## Solution 1

1. Images are classified into defect and non-defect (binary classification) using Xception CNN

2. Defect images are passed on to segmentation model implemented using ResUNet for the localization of defect

3. Output images are defect images with segmentation mask  


## Solution 2

1. Images are classified into defect type 0 (non-defect),1,2,3 and 4 (multilabel classification)using Vision Transformers

2. Four segmentation models have been have been trained (ResUNet), each for a single defect type. Based on the output of classification model, the image is passed to the respective segmentation model for the localization of defect

3. Output images are defect images with segmentation mask 


## References
1. [Custom data generator](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
2. [ResUNet](https://arxiv.org/pdf/1904.00592.pdf)
3. [Vision Transformers](https://arxiv.org/pdf/2010.11929.pdf)
4. [Coding ResNet in Keras](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33)
5. [Kaggle kernel]( https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode)
