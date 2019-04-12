
# ifndef caffexxx
# define caffexxx
#include<caffe/proto/caffe.pb.h>

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <caffe/caffe.hpp>
#include <sstream>
#include <stdlib.h>

#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/common.hpp"  
#include "caffe/layers/input_layer.hpp"  
#include "caffe/layers/inner_product_layer.hpp"  
#include "caffe/layers/dropout_layer.hpp"  
#include "caffe/layers/conv_layer.hpp"  
#include "caffe/layers/relu_layer.hpp"  

#include "caffe/layers/pooling_layer.hpp"  
#include "caffe/layers/lrn_layer.hpp"  
#include "caffe/layers/softmax_layer.hpp"  
#include "caffe/layers/prelu_layer.hpp" 

#include "caffe/layers/batch_norm_layer.hpp" 
#include <caffe/layers/scale_layer.hpp>
#include <caffe/layers/bias_layer.hpp>
#include <caffe/layers/reshape_layer.hpp>
#include <caffe/layers/upsample_layer.hpp>

//#include "prelu_layer.h"

namespace caffe
{

	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(LRNLayer);
	REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTER_LAYER_CLASS(Softmax);
	extern INSTANTIATE_CLASS(PReLULayer);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(ReshapeLayer);
	extern INSTANTIATE_CLASS(UpsampleLayer);
	//extern INSTANTIATE_CLASS(ReshapeLayer);
	//REGISTER_LAYER_CLASS(Reshape);
	//REGISTER_LAYER_CLASS(PReLU);
}
//
//using caffe::Blob;
using caffe::Caffe;
//using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;

using namespace cv;
using namespace std;
#endif
