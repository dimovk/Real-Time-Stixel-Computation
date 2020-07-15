#include <iostream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "multilayer_stixel_world.h"
#include "semi_global_matching.h"

#include <sl/Camera.hpp>

using namespace std;
using namespace sl;

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v * (1.f - s);
	tab[2] = v * (1.f - s * h);
	tab[3] = v * (1.f - s * (1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

static cv::Scalar dispToColor(float disp, float maxdisp = 64, float offset = 0)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp + offset, maxdisp) / maxdisp);
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
	cv::rectangle(img, cv::Rect(tl, br), cv::Scalar(255, 255, 255), 1);
}

class SGMWrapper
{

public:

	SGMWrapper(int numDisparities)
	{
		SemiGlobalMatching::Parameters param;
		param.numDisparities = numDisparities / 2;
		param.max12Diff = -1;
		param.medianKernelSize = -1;
		sgm_ = cv::Ptr<SemiGlobalMatching>(new SemiGlobalMatching(param));
	}

	void compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1)
	{
		cv::pyrDown(I1, I1_);
		cv::pyrDown(I2, I2_);

		sgm_->compute(I1_, I2_, D1_, D2_);

		cv::resize(D1_, D1, I1.size(), 0, 0, cv::INTER_CUBIC);
		cv::resize(D2_, D2, I1.size(), 0, 0, cv::INTER_CUBIC);
		D1 *= 2;
		D2 *= 2;
		cv::medianBlur(D1, D1, 3);
		cv::medianBlur(D2, D2, 3);
		SemiGlobalMatching::LRConsistencyCheck(D1, D2, 5);
	}

private:
	cv::Mat I1_, I2_, D1_, D2_, D2;
	cv::Ptr<SemiGlobalMatching> sgm_;
};

// Conversion function between sl::Mat and cv::Mat
cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}

int main(int argc, char* argv[])
{
	// Create a ZED camera object
    sl::Camera zed;

	// Set configuration parameters for the ZED
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD720;
	init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE;
	
	 // Open the camera
    sl::ERROR_CODE state = zed.open(init_parameters);
    if (state != ERROR_CODE::SUCCESS) {
        cout << "Error " << state << ", exit program." << endl;
        return EXIT_FAILURE;
    }

	// stereo SGBM
	const int numDisparities = 128;
	SGMWrapper sgm(numDisparities);
	
	// read camera parameters
	const cv::FileStorage fs("/home/dimovk/Desktop/stixels-live/camera.xml", cv::FileStorage::FORMAT_XML);
	CV_Assert(fs.isOpened());

	// input parameters
	MultiLayerStixelWorld::Parameters param;
	param.camera.fu = fs["FocalLengthX"];
	param.camera.fv = fs["FocalLengthY"];
	param.camera.u0 = fs["CenterX"];
	param.camera.v0 = fs["CenterY"];
	param.camera.baseline = fs["BaseLine"];
	param.camera.height = fs["Height"];
	param.camera.tilt = fs["Tilt"];
	param.dmax = numDisparities;

	cv::Mat disparity;
	MultiLayerStixelWorld stixelWorld(param);

	sl::Mat frame, im_left, im_right;

	while(1){

		state = zed.grab();

		if (state != ERROR_CODE::SUCCESS) {
        	cout << "Error " << state << ", exit program." << endl;
        	return EXIT_FAILURE;
    	}

		zed.retrieveImage(im_left, VIEW::LEFT);
		zed.retrieveImage(im_right, VIEW::RIGHT);

		cv::Mat fr_left = slMat2cvMat(im_left);
		cv::Mat fr_right = slMat2cvMat(im_right);

		cv::Mat fr_left_gr, fr_right_gr;
		cv::cvtColor(fr_left, fr_left_gr, cv::COLOR_BGR2GRAY);
		cv::cvtColor(fr_right, fr_right_gr, cv::COLOR_BGR2GRAY);

		cv::Mat I1 = fr_left_gr;
		cv::Mat I2 = fr_right_gr;

		if (I1.empty() || I2.empty()){
			std::cerr << "imread failed." << std::endl;
			break;
		}

		CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());
		CV_Assert(I1.type() == CV_8U || I1.type() == CV_16U);

		if (I1.type() == CV_16U){
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
			cv::normalize(I2, I2, 0, 255, cv::NORM_MINMAX);
			I1.convertTo(I1, CV_8U);
			I2.convertTo(I2, CV_8U);
		}

		const auto t1 = std::chrono::steady_clock::now();

		// compute dispaliry
		sgm.compute(I1, I2, disparity);
		disparity.convertTo(disparity, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);

		// compute stixels
		const auto t2 = std::chrono::steady_clock::now();

		std::vector<Stixel> stixels;
		stixelWorld.compute(disparity, stixels);

		const auto t3 = std::chrono::steady_clock::now();
		const auto duration12 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

		// colorize disparity
		cv::Mat disparityColor;
		disparity.convertTo(disparityColor, CV_8U, 255. / numDisparities);
		cv::applyColorMap(disparityColor, disparityColor, cv::COLORMAP_JET);
		disparityColor.setTo(cv::Scalar::all(0), disparity < 0);

		// put processing time
		cv::putText(disparityColor, cv::format("dispaliry computation time: %4.1f [msec]", 1e-3 * duration12),
			cv::Point(100, 50), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::putText(disparityColor, cv::format("stixel computation time: %4.1f [msec]", 1e-3 * duration23),
			cv::Point(100, 80), 2, 0.75, cv::Scalar(255, 255, 255));

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(I1, draw, cv::COLOR_GRAY2BGR);

		cv::Mat stixelImg = cv::Mat::zeros(I1.size(), draw.type());
		for (const auto& stixel : stixels)
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp));
		cv::addWeighted(draw, 1, stixelImg, 0.5, 0, draw);

		cv::imshow("disparity", disparityColor);
		cv::imshow("stixels", draw);

		const char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
	}

	// Close the camera
	zed.close();

	return 0; 
}
