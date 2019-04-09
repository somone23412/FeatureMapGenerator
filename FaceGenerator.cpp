#include "stdafx.h"
#include "FaceGenerator.h"
//for caffe layer registe
#include "caffexxx.h"


#define FGDEBUG
//#define SHOW_FEATURE


FaceGenerator::FaceGenerator(std::string modelFile, std::string trainedFile) {
	this->net = new caffe::Net<float>(modelFile, caffe::TEST);
	this->net->CopyTrainedLayersFrom(trainedFile);

	this->input_layer = this->net->input_blobs()[0];

	this->scale = 0.0078125;
	this->meanValue = { 127.5, 127.5, 127.5 };
}


FaceGenerator::~FaceGenerator() {
	delete this->net;
}


std::vector<cv::Mat> FaceGenerator::generateFace(cv::Mat img, std::vector<std::string> &layerNames) {
	//clean featureMaps
	this->featureMaps.clear();

	//set Params
	caffe::TransformationParameter tp;
	tp.set_scale(this->scale);
	tp.add_mean_value(this->meanValue[0]);
	tp.add_mean_value(this->meanValue[1]);
	tp.add_mean_value(this->meanValue[2]);

	//trans Mat to Caffe-Need-Type (#define USE_OPENCV)
	caffe::DataTransformer<float> dt(tp, caffe::Phase::TEST);
	cv::Mat tImg;
	//resize to input layer size
	cv::resize(img, tImg, cv::Size(input_layer->width(), input_layer->height()));
	dt.Transform(tImg, input_layer);
	
	//feature extract
	this->net->ForwardFrom(0);

	//trans each layer's feature to Mat
	for (int i = 0; i < layerNames.size(); i++) {
		caffe::Blob<float> *tmpLayer = net->blob_by_name(layerNames[i]).get();
		const float *begin = tmpLayer->cpu_data();
		const int length = tmpLayer->channels() * tmpLayer->width() * tmpLayer->height();
		float *feat = new float[length];

#ifdef FGDEBUG
		std::cout << "layer" << " : " << i << std::endl;
		std::cout << "channels" << " : " << tmpLayer->channels() << std::endl;
		std::cout << "width" << " : " << tmpLayer->width() << std::endl;
		std::cout << "height" << " : " << tmpLayer->height() << std::endl;
		std::cout << "data length" << " : " << length << std::endl;
#endif

		const float *end = tmpLayer->cpu_data() + length;
		::memcpy(feat, begin, sizeof(float) * length);

#ifdef SHOW_FEATURE
		//print feature
		for (int i = 0; i < length; i++) {
			std::cout << i << ":" << feat[i] << " ";
			if (0 == (i + 1) % 5 || length - 1 == i) {
				std::cout << std::endl;
			}
		}
#endif

		//trans result to Mat
		cv::Mat dst = transToMat(feat, length, tmpLayer);
		this->featureMaps.push_back(dst);
		delete feat;
	}
	return this->featureMaps;
}


std::vector<cv::Mat> FaceGenerator::generateFace(std::string imgPath, std::vector<std::string> &layerNames){
	cv::Mat img = cv::imread(imgPath);
	return this->generateFace(img, layerNames);
}


cv::Mat FaceGenerator::transToMat(float* feat, int length, caffe::Blob<float> *layer) {

	//trans float32* to Mat
	//data index in memeory : (n * K + k) * H + h) * W + w
	//n : img numbers; k : channels; h : height, w:width;
	//if channels > 3 then set K = 3
	int K = layer->channels() > 3 ? 3 : layer->channels();
	int H = layer->height();
	int W = layer->width();

	cv::Mat res = cv::Mat::zeros(H, W, CV_8UC3);

	for (int k = 0; k < K; k++){
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				//trans float to 0-255
				int val = (int)(*(feat + (k * H + h) * W + w) * 72.5 + 125.0);
				res.at<cv::Vec3b>(h, w)[k] = val;
			}
		}
	}

	return res;
}


void FaceGenerator::setMeanValue(std::vector<float> &meanValue) {
	this->meanValue = meanValue;
}


void FaceGenerator::setScale(float scale) {
	this->scale = scale;
}


std::vector<float> FaceGenerator::getMeanValue() const {
	return this->meanValue;
}


float FaceGenerator::getScale() const {
	return this->scale;
}

std::vector<cv::Mat> FaceGenerator::getFeatureMaps() const {
	return this->featureMaps;
}