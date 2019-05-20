//#include "stdafx.h"
#include "FeatureMapGenerator.h"


#define FGDEBUG
//#define SHOW_FEATURE


FeatureMapGenerator::FeatureMapGenerator(std::string modelFile, std::string trainedFile) {
	this->net = new caffe::Net<float>(modelFile, caffe::TEST);
	this->net->CopyTrainedLayersFrom(trainedFile);

	this->input_layer = this->net->input_blobs()[0];

	this->scale = 0.0078125;

	this->meanValue = { 127.5, 127.5, 127.5 };
}


FeatureMapGenerator::~FeatureMapGenerator() {
	delete this->net;
}


std::unordered_map<std::string, std::vector<cv::Mat>> FeatureMapGenerator::generateFeatureMaps(cv::Mat img, std::vector<std::string> &layerNames) {
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
		std::cout << "[layer" << " : " << layerNames[i] << "]"<<std::endl;
		std::cout << "channels" << " : " << tmpLayer->channels();
		std::cout << " | width" << " : " << tmpLayer->width();
		std::cout << " | height" << " : " << tmpLayer->height();
		std::cout << " | data length" << " : " << length << std::endl << std::endl;
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
		std::vector<cv::Mat> dst = transToMat(feat, tmpLayer);
		this->featureMaps.insert({ layerNames[i], dst });

		delete feat;
	}
	return this->featureMaps;
}


std::unordered_map<std::string, std::vector<cv::Mat>> FeatureMapGenerator::generateFeatureMaps(std::string imgPath, std::vector<std::string> &layerNames){
	cv::Mat img = cv::imread(imgPath);
	return this->generateFeatureMaps(img, layerNames);
}


std::vector<cv::Mat> FeatureMapGenerator::transToMat(float* feat, caffe::Blob<float> *layer) {

	//trans float32* to Mat
	//data index in memeory : (n * K + k) * H + h) * W + w
	//n : img numbers; k : channels; h : height, w:width;
	int K = layer->channels();
	int H = layer->height();
	int W = layer->width();

	std::vector<cv::Mat> res;

	for (int k = 0; k < K; k++){
		//save each Channel's featureMap
		cv::Mat tmp = cv::Mat::zeros(H, W, CV_8UC1);
		for (int h = 0; h < H; h++) {
			for (int w = 0; w < W; w++) {
				//trans float to 0-255
				float fea = *(feat + (k * H + h) * W + w);
				int val = (int)((fea + 1) * 127.5);
				if (val > 255) val = 255;
				if (val <= 0) val = 0;
				tmp.at<uchar>(h, w) = val;
			}
		}
		res.push_back(tmp);
	}

	if (3 == K){
		//save 3-Channel featureMap Singly
		cv::Mat tmp = cv::Mat::zeros(H, W, CV_8UC3);
		for (int k = 0; k < K; k++){
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {
					float fea = *(feat + (k * H + h) * W + w);
					int val = (int)((fea + 1) * 127.5);
					if (val > 255) val = 255;
					if (val < 0) val = 0;
					//OpenCV's default channels is bgr and what we needs is rgb
					tmp.at<cv::Vec3b>(h, w)[2 - k] = val;
				}
			}
		}
		res.push_back(tmp);
	}

	return res;
}


void FeatureMapGenerator::setMeanValue(std::vector<float> &meanValue) {
	this->meanValue = meanValue;
}


void FeatureMapGenerator::setScale(float scale) {
	this->scale = scale;
}


std::vector<float> FeatureMapGenerator::getMeanValue() const {
	return this->meanValue;
}


float FeatureMapGenerator::getScale() const {
	return this->scale;
}

std::unordered_map<std::string, std::vector<cv::Mat>> FeatureMapGenerator::getFeatureMaps() const {
	return this->featureMaps;
}