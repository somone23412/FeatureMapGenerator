# FeatureMapGenerator

[Getting start](#Getting-start)

[Documentation](#Documentation)

A little **c++ class** for getting the feature maps in any layer of the neural network by loading the trained **caffe** model.

By using this class you can see the **feature maps** of each layer in the neural network.

+ Here's an example running on [G_Model](https://github.com/Yijunmaverick/GenerativeFaceCompletion):

![featureMap](https://github.com/somone23412/FaceGenerator/blob/master/image/featureMap.jpg)

This example is mainly used to show what the generated feature map looks like. If you are interested in what this neural network has done, you can take a look at the model author's [matlab code](https://github.com/Yijunmaverick/GenerativeFaceCompletion/tree/master/matlab/FaceCompletion_testing).

# Getting start

You need [caffe](https://github.com/BVLC/caffe/) and [opencv](https://github.com/opencv/opencv) environments to use this class.

+ using example:

```cpp
#include "FeatureMapGenerator.h"

	...
	FeatureMapGenerator *featureMapGenerator = new FeatureMapGenerator("model/Model_G.prototxt", "model/Model_G.caffemodel");
	std::vector<std::string> layerNames = {"data","conv0_1", "conv1_1_new", "conv_decode1_1_new", "reconstruction_new"};
	
	cv::Mat faceImg;
	faceImg = cv::imread("faceimg/182701.png");
	featureMapGenerator->generateFeatureMaps(faceImg, layerNames);
	auto genImgs = featureMapGenerator->getFeatureMaps(); //"auto" here = std::unordered_map<std::string, std::vector<cv::Mat>>
	//if feature channels == 3, You can get a three-channel RGB Mat object in hashMap[name][3]
	cv::imshow("genImg", genImgs["data"][3]);
	delete featureMapGenerator;
	...
	
	
```

# Documentation
FeatureMapGenerator class Documentation.

## Contents

+ [PublicTypes](#PublicTypes)
+ [PublicFunctions](#PublicFunctions)

## FeatureMapGenerator Class
The [FeatureMapGenerator](#FeatureMapGenerator-class) class is used to getting the feature maps in any layer of the neural network by loading the trained caffe model.

**Header:** \#include "FeatureMapGenerator.h"

**Included Header:** \<opencv2\opencv.hpp\>, \<caffe\caffe.hpp\>, \<caffe\proto\caffe.pb.h\>, \<caffe\data_transformer.hpp\>, \<vector\>, \<string\>, \<unordered_map\>

**Inherits:** None

### PublicTypes

None

### PublicFunctions

| returns | functions |
| --- | --- |
||[FeatureMapGenerator](#func_1)(std::string modelFile, std::string trainedFile)|
||[~FeatureMapGenerator](#func_2)()|
|void|[setMeanValue](#func_3)(std::vector\<float\> &meanValue)|
|void|[setScale](#func_4)(float scale)|
|std::vector\<float\>|[getMeanValue](#func_5)() const|
|float| [getScale](#func_6)() const|
|std::unordered_map\<std::string, std::vector\<cv::Mat\>\>|[getFeatureMaps](#func_7)() const|
|std::unordered_map\<std::string, std::vector\<cv::Mat\>\>|[generateFeatureMaps](#func_8)(cv::Mat img, std::vector\<std::string\> &layerNames)|
|std::unordered_map\<std::string, std::vector\<cv::Mat\>\>|[generateFeatureMaps](#func_9)(std::string imgPath, std::vector\<std::string\> &layerNames)|

### Member Function Documentation

---

###### func_1

### FeatureMapGenerator::[FeatureMapGenerator](#func_1)(std::string modelFile, std::string trainedFile)

Constructs a [FeatureMapGenerator](#FeatureMapGenerator-class) and loads the trained caffe model.

---

###### func_2

### FeatureMapGenerator::[~FeatureMapGenerator](#func_2)()

Destroys the generator.

---

###### func_3

### void FeatureMapGenerator::[setMeanValue](#func_3)(std::vector<float> &meanValue) 

Sets the meanValue to *meanValue*.

**See also**  [getMeanValue()](#func_5).

---

###### func_4

### void FeatureMapGenerator::[setScale](#func_4)(float scale)

Sets the scale to *scale*.

**See also**  [getScale()](#func_6).

---

###### func_5

### std::vector<float> FeatureMapGenerator::[getMeanValue](#func_5)() const

returns the meanValue.

**See also** [setMeanValue()](#func_3).

---

###### func_6

### float FeatureMapGenerator::[getScale](#func_6)() const

returns the scale.

**See also**  [setScale()](#func_4).

---

###### func_7

### std::unordered_map\<std::string, std::vector\<cv::Mat\>\> FeatureMapGenerator::[getFeatureMaps](#func_7)() const 

returns the featureMaps.

---

###### func_8

### std::unordered_map\<std::string, std::vector\<cv::Mat\>\> FeatureMapGenerator::[generateFeatureMaps](#func_8)(cv::Mat img, std::vector\<std::string\> &layerNames)

+1 overloads.

Get the img from cv::Mat and forward the neural network, save the feature map of each layer to featureMaps.

returns the featureMaps.

**See also** [generateFeatureMaps()](#func_9).

###### func_9

---

### std::unordered_map\<std::string, std::vector\<cv::Mat\>\> FeatureMapGenerator::[generateFeatureMaps](#func_9)(std::string imgPath, std::vector\<std::string\> &layerNames)

+1 overloads.

Get the img from file and forward the neural network, save the feature map of each layer to featureMaps.

returns the featureMaps.

**See also** [generateFeatureMaps()](#func_8).
