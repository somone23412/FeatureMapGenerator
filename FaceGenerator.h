#pragma once

#include <opencv2\opencv.hpp>

#include <caffe\caffe.hpp>
#include <caffe\proto\caffe.pb.h>
#include <caffe\data_transformer.hpp>

#include <vector>
#include <string>


class FaceGenerator {
public:
	FaceGenerator(std::string modelFile, std::string trainedFile);
	~FaceGenerator();

	void setMeanValue(std::vector<float> &meanValue);
	void setScale(float scale);

	std::vector<float> getMeanValue() const;
	float getScale() const;
	std::vector<cv::Mat> getFeatureMaps() const;

	std::vector<cv::Mat> generateFace(cv::Mat img, std::vector<std::string> &layerNames);
	std::vector<cv::Mat> generateFace(std::string imgPath, std::vector<std::string> &layerNames);

private:
	cv::Mat transToMat(float* feat, int length, caffe::Blob<float> *layer);

private:
	caffe::Net<float> *net;
	caffe::Blob<float> *input_layer;

	float scale;
	std::vector<float> meanValue;

	std::vector<std::string> layerNames;
	std::vector<cv::Mat> featureMaps;
};

