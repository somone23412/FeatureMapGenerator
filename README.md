# FaceGenerator

+ This is a c++ class for face generation that is currently **under development**.
+ The goal is to transform the output features into face images by loading the trained **caffe** neural network model.
+ You need [caffe](https://github.com/BVLC/caffe/) and [opencv](https://github.com/opencv/opencv) environments to use this class.

code example:

```cpp
#include "FaceGenerator.h"

	...
	
	//set up faceGenerator
	FaceGenerator *faceGenerator = new FaceGenerator("model/model_v2-r50-symbol.prototxt", "model/model_v2-r50-symbol.caffemodel");

	std::vector<std::string> layerNames = { "data", "stage3_unit1_conv1", "stage3_unit1_conv2" };
	
	//generate img via cv::Mat
	cv::Mat faceImg, genImg;
	faceImg = cv::imread("faceimg/face.jpg");
	cv::imshow("faceImg", faceImg);

	genImg = faceGenerator->generateFace(faceImg, layerNames)[0];
	imshow("genImg", genImg);

	//generate img via abs-path
	std::vector<cv::Mat> genImgs;
	genImgs = faceGenerator->generateFace("faceimg/face2.jpg", layerNames);
	for (int i = 0; i < genImgs.size(); i++) {
		imshow("gen " + layerNames[i], genImgs[i]);
	}
	
	...
	
	
```

result:

![featureMap](https://github.com/somone23412/FaceGenerator/blob/master/image/featureMap.jpg)
