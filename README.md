# FaceGenerator

A **c++ class** for face generation that is currently **under development**.

By using this class you can see the **feature map** of each layer in the neural network.

+ Here's an example running on [G_Model](https://github.com/Yijunmaverick/GenerativeFaceCompletion):

![featureMap](https://github.com/somone23412/FaceGenerator/blob/master/image/featureMap.jpg)

If you are interested in how this example works, you can take a look at the model author's [matlab code](https://github.com/Yijunmaverick/GenerativeFaceCompletion/tree/master/matlab/FaceCompletion_testing), I just reproduce it in c++.

# How to get started

The goal is to transform the output features into images by loading the trained **caffe** neural network model.

You need [caffe](https://github.com/BVLC/caffe/) and [opencv](https://github.com/opencv/opencv) environments to use this class.

+ using example:

```cpp
#include "FaceGenerator.h"

	...
	FaceGenerator *faceGenerator = new FaceGenerator("model/Model_G.prototxt", "model/Model_G.caffemodel");
	std::vector<std::string> layerNames = {"data","conv0_1", "conv1_1_new", "conv_decode1_1_new", "reconstruction_new"};
	
	cv::Mat faceImg, inputImg, genImg;
	faceImg = cv::imread("faceimg/182701.png");
	faceGenerator->generateFace(faceImg, layerNames);
	auto genImgs = faceGenerator->getFeatureMaps(); //auto here = std::unordered_map<std::string, std::vector<cv::Mat>>
	//if feature channels == 3, You can get a three-channel RGB Mat object in hashMap[name][3]
	cv::imshow("genImg", genImgs["data"][3]);
	...
	
	
```
