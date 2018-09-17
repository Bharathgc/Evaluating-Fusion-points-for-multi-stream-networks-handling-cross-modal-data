## Evaluating Fusion points for Multi-Stream Networks handling cross modal data

Object detection using RGB-D images has become a trending topic these days due to its numerous applications. This project uses RGB and Depth images as input into two different convolutional network of same architecture (namely VGGNet, RESNet, AlexNet) and fuses them in deeper layers to improves the class prediction performance with respect to various metrics like run-time, number of parameters and accuracy. Our approach compares the different possible fusion points in a network to come up with the best tradeoff between complexity and prediction. 

<p align="center">
<img src="https://github.com/Bharathgc/Evaluating-Fusion-points-for-multi-stream-networks-handling-cross-modal-data/blob/master/Updated%20Code/Capture2.PNG" />
</p>

## Dataset:

We experimented with NYUD V2 dataset which is a collection of video sequences from various indoor scenes recorded by both RGB and depth cameras from Microsoft Kinect. To balance the number of examples in each class, the images are shifted in space or inverted vertically and/or horizontally adding some noise to generate new images for each class. 

