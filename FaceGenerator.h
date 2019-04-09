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

	int generateFace(cv::Mat img, cv::Mat &dst);
	int generateFace(std::string imgPath, cv::Mat &dst);

private:
	cv::Mat transToMat(float feat[], int length);

private:
	caffe::Net<float> *net;
	std::vector<cv::Mat> featureMaps;
	float scale;
	std::vector<float> meanValue;

};

