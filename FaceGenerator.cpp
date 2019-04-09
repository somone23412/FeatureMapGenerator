#include "stdafx.h"
#include "FaceGenerator.h"
//for caffe layer registe
#include "caffexxx.h"


#define FGDEBUG


FaceGenerator::FaceGenerator(std::string modelFile, std::string trainedFile) {
	this->net = new caffe::Net<float>(modelFile, caffe::TEST);
	this->net->CopyTrainedLayersFrom(trainedFile);

	this->scale = 0.0078125;
	this->meanValue = { 127.5, 127.5, 127.5 };
}


FaceGenerator::~FaceGenerator() {
	delete this->net;
}


int FaceGenerator::generateFace(cv::Mat img, cv::Mat &dst) {

	caffe::Blob<float> *input_layer = this->net->input_blobs()[0];
	caffe::Blob<float> *out_layer = this->net->output_blobs()[0];

	//set Params
	caffe::TransformationParameter tp;
	tp.set_scale(this->scale);
	tp.add_mean_value(this->meanValue[0]);
	tp.add_mean_value(this->meanValue[1]);
	tp.add_mean_value(this->meanValue[2]);
	

	//trans Mat to Caffe-Need-Type (#define USE_OPENCV)
	caffe::DataTransformer<float> dt(tp, caffe::Phase::TEST);
	cv::Mat tImg;
	cv::resize(img, tImg, cv::Size(input_layer->width(), input_layer->height()));
	dt.Transform(tImg, input_layer);
	
	//feature extract
	this->net->ForwardFrom(0);

	//record result
	float *feat = new float[out_layer->channels()];
	const float *begin = out_layer->cpu_data();
	const float *end = out_layer->cpu_data() + out_layer->channels();
	::memcpy(feat, begin, sizeof(float) * out_layer->channels());

#ifdef FGDEBUG

	//print feature
	for (int i = 0; i < out_layer->channels(); i++) {
		std::cout << i << ":" << feat[i] << " ";
		if (0 == (i + 1) % 5 || out_layer->channels() - 1 == i) {
			std::cout << std::endl;
		}
	}

#endif

	//trans result to Mat
	dst = transToMat(feat, out_layer->channels());

	delete feat;

	//return 1 if failed to trans result to Mat
	if ((dst.size <= 0) || (dst.rows <= 0))
		return 1;
	return 0;
}


int FaceGenerator::generateFace(std::string imgPath, cv::Mat &dst){
	cv::Mat img = cv::imread(imgPath);
	return this->generateFace(img, dst);
}


cv::Mat FaceGenerator::transToMat(float* feat, int length) {


	//trans float32[][] to Mat
	int h = 224;
	int w = 224;

	cv::Mat res = cv::Mat::zeros(h, w, CV_8UC3);

#ifdef FGDEBUG

	//for example, 512 = 16 * 32 here
	h = 16;
	w = 32;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int val = (int)(*(feat + i * 16 + j) * 72.5 + 125.0);
			//std::cout << "[" << i << "," << j << "] "<< val << std::endl;
			res.at<cv::Vec3b>(i, j)[0] = 0;
			res.at<cv::Vec3b>(i, j)[1] = val;
			res.at<cv::Vec3b>(i, j)[2] = 0;
		}
	}

#endif



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