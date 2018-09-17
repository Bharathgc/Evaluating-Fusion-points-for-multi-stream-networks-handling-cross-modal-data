## Evaluating Fusion points for Multi-Stream Networks handling cross modal data

Object detection using RGB-D images has become a trending topic these days due to its numerous applications. This project uses RGB and Depth images as input into two different convolutional network of same architecture (namely VGGNet, RESNet, AlexNet) and fuses them in deeper layers to improves the class prediction performance with respect to various metrics like run-time, number of parameters and accuracy. Our approach compares the different possible fusion points in a network to come up with the best tradeoff between complexity and prediction. 

<p align="center">
<img src="https://github.com/Bharathgc/Evaluating-Fusion-points-for-multi-stream-networks-handling-cross-modal-data/blob/master/Updated%20Code/Capture2.PNG" />
</p>

### Dataset:

We experimented with NYUD V2 dataset which is a collection of video sequences from various indoor scenes recorded by both RGB and depth cameras from Microsoft Kinect. To balance the number of examples in each class, the images are shifted in space or inverted vertically and/or horizontally adding some noise to generate new images for each class. 

### Pre-Requisites

- Python 3.0 or higher
- Tensorflow (Runs better on Tensorflow-gpu)
- Opencv
- Numpy
- Matplotlib

### Installing

Download or `git clone` the repoaitory to local machine. Change the input directory location in the `fused_classifier.py` file 

### Running The tests

To execute a fusion point test, Change to corresponding function name at `line 208` in the fusedClassifier.py file

1. Alexnet
  - Fuse Points ==> Name of the function
	  - Fusion at 2	==>	alexnet_fused2
	  - Fusion at 3	==>	alexnet_fused3
	  - Fusion at 4	==>	alexnet_fused4
	  - Fusion at 5	==>	alexnet_fused5
	  - Fusion at 6	==>	alexnet_fused6
2. VGGnet
  - Fuse points ==>  Name of the function
	  - Fusion at 2 ==> vggnet_fused2
	  - Fusion at 3	==>	vggnet_fused3
	  - Fusion at 4	==>	vggnet_fused4
	  - Fusion at 5	==>	vggnet_fused5
	  - Fusion at 6	==>	vggnet_fused6
3. Resnet
  - Fuse points ==>  Name of the function
	  - Fusion at 2 ==> resnet_fused2
	  - Fusion at 3	==>	resnet_fused3
	  - Fusion at 4	==>	resnet_fused4
	  - Fusion at 5	==>	resnet_fused5
	  - Fusion at 6	==>	resnet_fused6

### Contributing Authors

1. Kausic Gunasekkar - [Profile](https://github.com/kausic94)

### License

This project is licensed under the MIT License - see the [LICENSE.md]() file for details
