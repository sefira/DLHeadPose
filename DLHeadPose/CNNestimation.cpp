#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "tiny_cnn.h"

//#define VISUALIZE
//#define FOR_VIDEO

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;

cv::VideoCapture m_videocapture;
static const std::string weights_filename = "xbu-weights_282";
network<mse, adagrad> nn; 

int ColouringWeightBias()
{
	for (int i = 0; i < nn.depth(); i++)
	{
		vec_t& weight = nn[i]->weight();
		vec_t& bias = nn[i]->bias();

		for (int j = 0; j < weight.size(); j++)
		{
			weight[j] = i * 2;
		}
		for (int j = 0; j < bias.size(); j++)
		{
			bias[j] = i * 2 + 1;
		}
	}
	ofstream ofs("Colouring-weights");
	ofs << nn;

	return 0;
}

// convert tiny_cnn::image to cv::Mat and resize
cv::Mat Image2Mat(image<>& img) {
	cv::Mat ori(img.height(), img.width(), CV_8U, &img.at(0, 0));
	cv::Mat resized;
	cv::resize(ori, resized, cv::Size(), 3, 3, cv::INTER_AREA);
	return resized;
}

int ResizeImage(cv::Mat img, double minv, double maxv,
	int w,
	int h,
	vec_t& data)
{
	cv::Mat_<uint8_t> resized;
	cv::resize(img, resized, cv::Size(w, h));
	for (int i = 0; i < resized.size().width; i++)
	{
		for (int j = 0; j < resized.size().height; j++)
		{
			double temp = (resized[i][j]) * (maxv - minv) / 255.0 + minv;
			data.push_back(temp);
			//cout << temp << endl;
		}
	}

	return 0;
}

int GetImageDataFromVideo(vec_t& data)
{
	cv::Mat frame;
	m_videocapture >> frame;
	// cannot open, or it's not an image
	if (frame.data == nullptr)
	{
		return 0;
	}
	//convert a frame into gray
	cv::Mat cvtBGRimg;
	cv::cvtColor(frame, cvtBGRimg, CV_BGR2GRAY);

	//resize a image 
	ResizeImage(cvtBGRimg,- 1.0, 1.0, 32, 32, data);

	return 0;
}

int GetImageDataFromPicture(const std::string &picutrefilename, vec_t& data)
{
	auto img = cv::imread(picutrefilename, cv::IMREAD_GRAYSCALE);
	// cannot open, or it's not an image
	if (img.data == nullptr)
	{
		return 0;
	}
	ResizeImage(img, -1.0, 1.0, 32, 32, data);
	return 0;
}

int Recognize(const std::string& dictionary) {
	nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)
		<< average_pooling_layer<tan_h>(28, 28, 6, 2)
		<< convolutional_layer<tan_h>(14, 14, 5, 6, 16)
		<< average_pooling_layer<tan_h>(10, 10, 16, 2)
		<< convolutional_layer<tan_h>(5, 5, 5, 16, 120)
		//2 output which are pitch and yaw
		<< fully_connected_layer<tiny_cnn::activation::identity>(120, 2);

	// load nets
	ifstream ifs(weights_filename);
	ifs >> nn;

	//////////////////////////////////////////////////////
	///////////////for video//////////////////////////////
	//////////////////////////////////////////////////////
#ifdef FOR_VIDEO
	vec_t data;
	m_videocapture = cv::VideoCapture(0);
	while (1)
	{
		vec_t().swap(data);
		GetImageDataFromVideo(data);

		// recognize
		vec_t result = nn.predict(data);
		cout << result[0] << ' ' << result[1] << endl;
	}
#endif
	////////////////////////////////////////////////////////
	///////////////for picture//////////////////////////////
	////////////////////////////////////////////////////////
#ifndef FOR_VIDEO

	vec_t test[4];
	GetImageDataFromPicture("down_down\\image0\\1_1.jpg",test[0]);
	GetImageDataFromPicture("down_up\\image-30\\1_1.jpg",test[1]);
	GetImageDataFromPicture("up_down\\image45\\1_1.jpg",test[2]);
	GetImageDataFromPicture("up_up\\image60\\1_1.jpg", test[3]);

	vec_t result[4];

	for (int i = 0; i < 4; i++)
	{
		result[i] = nn.predict(test[i]);
		cout << result[i][0] << ' ' << result[i][1] << endl;
	}
	////////////////////////////////////////////////////////
	////////////////visualize///////////////////////////////
	////////////////////////////////////////////////////////
#ifdef VISUALIZE
	// visualize outputs of each layer
	for (size_t i = 0; i < nn.depth(); i++) {
		auto out_img = nn[i]->output_to_image();
		cv::imshow("layer:" + std::to_string(i), Image2Mat(out_img));
	}
	// visualize filter shape of first convolutional layer
	auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
	cv::imshow("weights:", Image2Mat(weight));
#endif

	cv::waitKey(0);
#endif
	return 0;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "please specify image file" << endl;;
		//return 0;
	}
	Recognize("xbu-weights");
	return 0;
}