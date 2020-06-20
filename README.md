# AlexNet-Implementation

Image classification of our own subset of Tiny ImageNet data by reproducing the state of the art convolutional neural network built by Alex Krizhevsky et al. called AlexNet.

**Goal of Model**:

This is an image classification problem. The goal of the model is to classify the test set as accurately as possible using images from the Tiny ImageNet dataset. 

## Data

A general description and download links of the ImageNet challenge and datasets can be seen here: https://tiny-imagenet.herokuapp.com/ and http://www.image-net.org/challenges/LSVRC/2014/

Tiny ImageNet is a subset of ImageNet containing 200 classes, each with 500 images. We further restrict the dataset by creating **'Tiny Image10'**. This new subset contains 10 labels with 500 images each. Similar to the proportions of Tiny ImageNet, 'Tiny Image10' contains 400 training images, 50 validation images, and 50 test images for each label.
An example of Tarantula and Tailed Frog labels respectively:

![Tarantula](https://imgur.com/N81Ol1a.png)
![Tailed Frog](https://imgur.com/ui4fT4P.png)

![Sample Result1](https://i.imgur.com/VWVz96P.png)
![Sample Result2](https://i.imgur.com/KUxk7Fh.png)

The rest of this project is completed using Pytorch.

## Model
### Preprocessing
In terms of preprocessing, the images were resized, cropped, and normalized. To normalize the images, the mean and start deviation of the pixels RGB data is used as provided by Krizhevsky et al (mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]). 

### Building Model
AlexNet is an eight layer CNN. Figure below exhibits the convolutional and pooling layers, as well as the three dense (fully-connected) layers that the input image passes through. One of the ways in which AlexNet distinguished itself from traditional CNN architectures was through its use of overlapping pooling. Pooling traditionally involved collecting outputs of neighboring neurons directly, with no overlap. Overlap was found to reduce the error rate and prevent overfitting. AlexNet also utilizes the ReLU activation function, which is a piecewise linear function that will return the input value if the summed weighted value is positive, otherwise it will output zero. ReLU has also been found to reduce training time in comparison to other common choices such as the hyperbolic tangent function. It is applied after each convolutional and fully connected layer. Dropout is applied before the first and the second fully connected layers to achieve regularization and reduce overfitting. No normalization layers are applied because the original paper found that normalization did not result in higher accuracy on test data with AlexNet. 

![model](https://i.imgur.com/Xr4mimQ.png)

### Gridsearching

![](https://i.imgur.com/hbqR4cI.png)
For each set of parameters, we train the model on the training data, retrieve its training, validation, and test metrics and then forwards that information for further analysis to Tensorboard. We ran 12 different parameters inputs with 35 epochs in each trial run, leading to a total run time of approximately 126 minutes. From the parameters discussed above, we Gridsearch AlexNet across the following values:  'batch size' : [12], learning rate : [0.0001, 0.001, 0.005], 'momentum' : [0.9, 0.5], and 'gamma' : [0.1, 0.01]. All of the values and results can be seen in the table below:
![table](https://i.imgur.com/zfJXelX.png)

## Results
More results on some test images can be seen below:

![](https://i.imgur.com/wlaNVOy.png)
![](https://i.imgur.com/V1YkuEV.png)

The green caption on the images indicate the model predicted correctly, and red, otherwise.

The figure below shows the models accuracy on each label across the entire test set. We can see that some labels performed way better than others, as can be seen in the 'goldfish' accuracy when compared to 'scorpion'.

![](https://i.imgur.com/mqGArr4.png)

## Rerunning Model

To rerun the model follow the steps below:
1. Run the functions in loading_data.ipynb to create metrics saved in /runs/*
1. Check Tensorboard for train and validation loss and accuracy to compare best parameters
1. Run visuals.ipynb to extract values from tensorboard files, and create visualizations

## References

> Jia Deng et al. “ImageNet: A large-scale hierarchical image database”.In:2009 IEEE Conference on Computer Vision and Pattern Recognition(2009).doi:10.1109/cvpr.2009.5206848.

> Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. “ImageNet”.In:Communications of the ACM60.6 (2017), pp. 84–90.doi:10.1145/3065386.


