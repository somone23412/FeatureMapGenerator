# FaceGenerator

+ This is a c++ class for face generation that is currently **under development**.
+ The goal is to transform the output features into face images by loading the trained **caffe** neural network model.
+ You need [caffe](https://github.com/BVLC/caffe/) and [opencv](https://github.com/opencv/opencv) environments to use this class.

code example:

```cpp
#include "FaceGenerator.h"

	...
	
	FaceGenerator *faceGenerator = new FaceGenerator("model.prototxt", "model.caffemodel");
	
	//read img
	cv::Mat faceImg, genImg, genImg2;
	faceImg = cv::imread("faceimg/face.jpg");
	assert((faceImg.cols > 0) && (faceImg.rows > 0));
	cv::imshow("faceImg", faceImg);

	//generate img via cv::Mat
	if (0 == faceGenerator->generateFace(faceImg, genImg)){
		std::cout << "Generate via cv::Mat successed!" << std::endl;
		imshow("genImg", genImg);
	}else {
		std::cout << "Generate via cv::Mat failed!" << std::endl;
	}

	//generate img via abs-path
	if (0 == faceGenerator->generateFace("faceimg/face2.jpg", genImg2)){
		std::cout << "Generate via abs-path successed!" << std::endl;
		imshow("genImg2", genImg2);
	}
	else {
		std::cout << "Generate via abs-path failed!" << std::endl;
	}
	
	...
	
	
```

