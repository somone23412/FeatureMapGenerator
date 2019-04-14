# FeatureMapGenerator

A little **c++ class** for getting the feature maps in any layer of the neural network by loading the trained **caffe** model.

By using this class you can see the **feature maps** of each layer in the neural network.

+ Here's an example running on [G_Model](https://github.com/Yijunmaverick/GenerativeFaceCompletion):

![featureMap](https://github.com/somone23412/FaceGenerator/blob/master/image/featureMap.jpg)

This example is mainly used to show what the generated feature map looks like. If you are interested in what this neural network has done, you can take a look at the model author's [matlab code](https://github.com/Yijunmaverick/GenerativeFaceCompletion/tree/master/matlab/FaceCompletion_testing).

# Getting start

You need [caffe](https://github.com/BVLC/caffe/) and [opencv](https://github.com/opencv/opencv) environments to use this class ("caffexxx.h" included in FceGenerator.cpp is only for caffe-layer register, not a part of this class indeed).

+ using example:

```cpp
#include "FeatureMapGenerator.h"

	...
	FeatureMapGenerator *featureMapGenerator = new FeatureMapGenerator("model/Model_G.prototxt", "model/Model_G.caffemodel");
	std::vector<std::string> layerNames = {"data","conv0_1", "conv1_1_new", "conv_decode1_1_new", "reconstruction_new"};
	
	cv::Mat faceImg;
	faceImg = cv::imread("faceimg/182701.png");
	featureMapGenerator->generateFace(faceImg, layerNames);
	auto genImgs = featureMapGenerator->getFeatureMaps(); //"auto" here = std::unordered_map<std::string, std::vector<cv::Mat>>
	//if feature channels == 3, You can get a three-channel RGB Mat object in hashMap[name][3]
	cv::imshow("genImg", genImgs["data"][3]);
	delete featureMapGenerator;
	...
	
	
```
