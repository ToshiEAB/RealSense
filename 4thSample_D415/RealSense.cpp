/*
Programmed by Toshikazu Kuroda at Aichi Bunkyo University

<Description>
This is a source code for the 4th sample program.
It works with RealSense D415 camera model only.
In this program, color and depth frames are aligned by using the left infrared camera as a color camera.

There are options for 2D tracking based on color, image subtraction, and background subtraction.
Terminate the program by pressing the 'q' key.

<References>
For color tracking & Image subtraction, see https://github.com/MicrocontrollersAndMore
For minimizing the infrared part of color-plus-infrared frame generated by the left infrared camera, 
see https://realsense.intel.com/wp-content/uploads/sites/63/BKMs-For-Tuning-RealSense_D4xx_Cameras_WP_1.7.pdf

<Acknowledgement>
A software developer "UnaNancyOwen" shared information about the use of left infrared camera for aligning color and depth frames as well as how to display a BGR frame using the infrared camera. 

<System requirements>
1) OS: Windows 10, Mac OS, or Linux Ubuntu 16.04 LTS
2) Hardware: USB3.0 port

*/


#include <librealsense2/rs.hpp> 
#include <librealsense2/rsutil.h> 
#include <librealsense2/rs_advanced_mode.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Blob
{
public:
	std::vector<cv::Point> contour;
	cv::Rect boundingRect;
	cv::Point centerPosition;
	double dblDiagonalSize;
	double dblAspectRatio;

	Blob(std::vector<cv::Point> _contour)
	{
		contour = _contour;
		boundingRect = cv::boundingRect(contour);
		centerPosition.x = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
		centerPosition.y = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;
		dblDiagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));
		dblAspectRatio = (float)boundingRect.width / (float)boundingRect.height;
	}
};

class RealSenseApp
{
private:
	// Classes
	rs2::pipeline pipe;
	rs2::config cfg;
	rs2::frameset frames;
	rs2::frame color_frame;
	rs2::frame depth_frame;
	rs2::pipeline_profile profile;

	// Containers for intrinsic and extrinsic parameters of color and depth cameras
	struct rs2_intrinsics intrin_depth;

	// Stream parameters
	const int COLOR_WIDTH = 1280;
	const int COLOR_HEIGHT = 720;
	const int COLOR_FPS = 30;
	const int DEPTH_WIDTH = 1280;
	const int DEPTH_HEIGHT = 720;
	const int DEPTH_FPS = 30;

	// Mat data containers
	cv::Mat colorFrame;
	cv::Mat colorFrame_copy;
	cv::Mat depthFrame;

	// Set color parameters
	const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
	const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
	const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
	const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
	const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

	// For editing frame
	cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

	// FPS
	std::chrono::system_clock::time_point start, end;
	int fps = 0;

	// Select a method for 2D tracking
	unsigned char trackingMethod = 0; // 0 = Color tracking, 1 = Image subtraction, 2 = Background subtraction; otherwise, no tracking

	// Memory
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> convexHulls;
	std::vector<Blob> blobs;
	std::vector<cv::Point> ColorPoints;

public:
	// Constructor
	RealSenseApp()
	{
		rs2_error* e = 0;
		rs2_context* ctx = rs2_create_context(RS2_API_VERSION, &e);
		rs2_device_list* dev_list = rs2_query_devices(ctx, &e);
		rs2_device* dev = rs2_create_device(dev_list, 0, &e);
		int is_advanced_mode_enabled;
		rs2_is_enabled(dev, &is_advanced_mode_enabled, &e);
		if (!is_advanced_mode_enabled)
		{
			rs2_toggle_advanced_mode(dev, 1, &e);
		}

		// Enable Color & Depth streams
		cfg.enable_stream(RS2_STREAM_INFRARED, COLOR_WIDTH, COLOR_HEIGHT, RS2_FORMAT_BGR8, COLOR_FPS);
		cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_Z16, DEPTH_FPS);
		
		// Start the pipeline for the streams
		profile = pipe.start(cfg);

		// Minimizes the infrared part of infrared-plus-color frame
		STColorCorrection ColorCorrection;
		ColorCorrection = { 0.520508f, 1.99023f, 1.50684f, -2.0f, -2.0f, -0.0820312f, 1.12305f, 1.01367f, 1.69824f, -2.0f, 0.575195f, -0.411133f };
		rs2_set_color_correction(dev, &ColorCorrection, &e);


		std::cout << "\n";

		// Get intrinsic depth/infrared camera
		rs2::video_stream_profile stream_depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
		intrin_depth = stream_depth.get_intrinsics();
		rs2_distortion model_depth = intrin_depth.model;
		std::cout << "Intrinsics: Depth/Infrared Camera\n";
		std::cout << "Principal Point         : " << intrin_depth.ppx << ", " << intrin_depth.ppy << "\n";
		std::cout << "Focal Length            : " << intrin_depth.fx << ", " << intrin_depth.fy << "\n";
		std::cout << "Distortion Model        : " << model_depth << "\n";
		std::cout << "Distortion Coefficients : [" << intrin_depth.coeffs[0] << "," << intrin_depth.coeffs[1] << "," <<
			intrin_depth.coeffs[2] << "," << intrin_depth.coeffs[3] << "," << intrin_depth.coeffs[4] << "]" << "\n";
		std::cout << "\n";
	}

	// Destructor
	~RealSenseApp()
	{
		// Terminate the pipeline
		pipe.stop();

		// Release memory
		std::vector<std::vector<cv::Point>>().swap(contours);
		std::vector<std::vector<cv::Point>>().swap(convexHulls);
		std::vector<Blob>().swap(blobs);
		std::vector<cv::Point>().swap(ColorPoints);

		// Close all OpenCV windows
		cv::destroyAllWindows();
	}

	void run()
	{
		cv::setUseOptimized(true);
		std::cout << "Terminate by pressing the q key\n";

		// Set the location of OpenCV windows	
		cv::namedWindow("Color Image");
		cv::moveWindow("Color Image", 50, 100);
		cv::namedWindow("Depth Image");
		cv::moveWindow("Depth Image", COLOR_WIDTH + 50, 100);

		// For color tracking
		cv::Mat hsvFrame, threshLow, threshHigh, resultFrame_hsv;

		// For the image-subtraction method
		cv::Mat colorFrame_current, colorFrame_previous, grayFrame_current, grayFrame_previous, diffFrame, resultFrame_imgsub;
		updateFrame();
		colorFrame_previous = colorFrame.clone();

		// For the background-subtraction method
		cv::Mat3f f_colorFrame, f_accumulatorFrame;
		//cv::Mat movingaverageFrame, graymovingaverageFrame;
		cv::Mat grayFrame, backgroundFrame, graybackgroundFrame, resultFrame_backsub;
		f_accumulatorFrame = cv::Mat::zeros(cv::Size(COLOR_WIDTH, COLOR_HEIGHT), CV_32FC3);

		// For each tracking method
		cv::Mat resultFrame;

		start = std::chrono::system_clock::now();

		while (1)
		{
			updateFrame();
			colorFrame_copy = colorFrame.clone();
			FPS();

			switch (trackingMethod)
			{
			case 0: // Color tracking (Detects reddish colors with the following parameters)
				cv::cvtColor(colorFrame, hsvFrame, CV_BGR2HSV);
				cv::inRange(hsvFrame, cv::Scalar(0, 155, 155), cv::Scalar(18, 255, 255), threshLow);
				cv::inRange(hsvFrame, cv::Scalar(165, 155, 155), cv::Scalar(179, 255, 255), threshHigh);
				cv::add(threshLow, threshHigh, resultFrame_hsv);

				// Optional edits for reducing noise
				cv::GaussianBlur(resultFrame_hsv, resultFrame_hsv, cv::Size(5, 5), 0);
				cv::dilate(resultFrame_hsv, resultFrame_hsv, structuringElement3x3);
				cv::erode(resultFrame_hsv, resultFrame_hsv, structuringElement3x3);

				cv::imshow("Color Tracking", resultFrame_hsv);

				resultFrame = resultFrame_hsv.clone();
				break;
			case 1: // Image subtraction
				colorFrame_current = colorFrame.clone();
				cv::cvtColor(colorFrame_current, grayFrame_current, CV_BGR2GRAY);
				cv::cvtColor(colorFrame_previous, grayFrame_previous, CV_BGR2GRAY);
				cv::GaussianBlur(grayFrame_current, grayFrame_current, cv::Size(5, 5), 0);
				cv::GaussianBlur(grayFrame_previous, grayFrame_previous, cv::Size(5, 5), 0);
				cv::absdiff(grayFrame_current, grayFrame_previous, diffFrame);
				cv::threshold(diffFrame, resultFrame_imgsub, 30, 255.0, CV_THRESH_BINARY);
				colorFrame_previous = colorFrame_current.clone();

				// Optional edits for reducing noise
				cv::dilate(resultFrame_imgsub, resultFrame_imgsub, structuringElement5x5);
				cv::erode(resultFrame_imgsub, resultFrame_imgsub, structuringElement5x5);

				cv::imshow("Image Subtraction", resultFrame_imgsub);

				resultFrame = resultFrame_imgsub.clone();
				break;
			case 2: // Background subtraction
				f_colorFrame = cv::Mat3f(colorFrame);
				cv::accumulateWeighted(f_colorFrame, f_accumulatorFrame, 0.05);
				cv::convertScaleAbs(f_accumulatorFrame, backgroundFrame);
				cv::imshow("Background Frame", backgroundFrame);

				cv::cvtColor(colorFrame, grayFrame, CV_BGR2GRAY);
				cv::cvtColor(backgroundFrame, graybackgroundFrame, CV_BGR2GRAY);
				cv::absdiff(grayFrame, graybackgroundFrame, diffFrame);
				cv::threshold(diffFrame, resultFrame_backsub, 30, 255.0, CV_THRESH_BINARY);

				// Optional edits for reducing noise
				cv::dilate(resultFrame_backsub, resultFrame_backsub, structuringElement5x5);
				cv::erode(resultFrame_backsub, resultFrame_backsub, structuringElement5x5);

				cv::imshow("Background Subtraction", resultFrame_backsub);

				resultFrame = resultFrame_backsub.clone();
				break;
			}

			if (trackingMethod < 3) DetectCenterOfObject(resultFrame);

			if (colorFrame_copy.rows != 0 && colorFrame_copy.cols != 0) cv::imshow("Color Image", colorFrame_copy);
			if (depthFrame.rows != 0 && depthFrame.cols != 0) cv::imshow("Depth Image", depthFrame);

			char c = cv::waitKey(1);
			if (c == 'q') break;
		}
	}

private:
	void updateFrame()
	{
		// Wait for next set of frames
		frames = pipe.wait_for_frames();

		// Get color & depth frames
		color_frame = frames.get_infrared_frame();
		depth_frame = frames.get_depth_frame();		

		// Convert the default format of the frames to Mat format
		colorFrame = cv::Mat(cv::Size(COLOR_WIDTH, COLOR_HEIGHT), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
		depthFrame = cv::Mat(cv::Size(DEPTH_WIDTH, DEPTH_HEIGHT), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

		// Transform 16-bit frame to 8-bit and then to BGR -- just for display purpose
		depthFrame.convertTo(depthFrame, CV_8UC1, -255.f/10000.f, 255.f);
		cv::cvtColor(depthFrame, depthFrame, CV_GRAY2BGR);		
	}

	void FPS()
	{
		// Calculate frames per second (fps) and show it on depth frame
		end = std::chrono::system_clock::now();
		fps = (int)(1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
		std::string fps_str = std::string(std::to_string(fps) + "fps").c_str();
		cv::putText(depthFrame, fps_str, cv::Point(2, 28), cv::FONT_HERSHEY_COMPLEX, 1.0, SCALAR_BLUE, 1, CV_AA);		
		start = std::chrono::system_clock::now();
	}

	void DetectCenterOfObject(cv::Mat& result)
	{
		// Convert cv::Mat to a vector of points			
		cv::findContours(result, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// Draw ConvexHulls using the vector of points
		convexHulls.resize(contours.size());
		for (unsigned int i = 0; i < contours.size(); i++) cv::convexHull(contours[i], convexHulls[i]);

		for (auto &convexHull : convexHulls)
		{
			Blob possibleBlob(convexHull);

			// The following parameters affect the sensitivity of detecting moving objects
			if (possibleBlob.boundingRect.area() > 100 &&
				possibleBlob.dblAspectRatio >= 0.2 &&
				possibleBlob.dblAspectRatio <= 1.2 &&
				possibleBlob.boundingRect.width > 50 &&
				possibleBlob.boundingRect.height > 50 &&
				possibleBlob.dblDiagonalSize > 30)
			{
				blobs.push_back(possibleBlob);
			}
		}

		convexHulls.clear();

		// ConvexHulls
		for (auto &blob : blobs) convexHulls.push_back(blob.contour);
		cv::Mat imgConvexHulls(result.size(), CV_8UC3, SCALAR_BLACK);
		cv::drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);
		//cv::imshow("ConvexHulls", imgConvexHulls);

		// The center of gravity & rectangle surrounding the detected part
		for (auto &blob : blobs)
		{
			cv::Rect selectedArea(blob.boundingRect.x, blob.boundingRect.y, blob.boundingRect.width, blob.boundingRect.height);
			cv::Mat Frame_selectedArea = result(selectedArea);

			// Get the center of gravity within a selected area
			cv::Moments mu = moments(Frame_selectedArea, true);
			int center_x = (int)(mu.m10 / mu.m00);
			int center_y = (int)(mu.m01 / mu.m00);

			int colorPoint_x = blob.boundingRect.x + center_x;
			int colorPoint_y = blob.boundingRect.y + center_y;

			// Show detected parts
			cv::Point pt = { colorPoint_x, colorPoint_y };
			cv::rectangle(colorFrame_copy, blob.boundingRect, SCALAR_RED, 1);
			cv::circle(colorFrame_copy, pt, 3, SCALAR_GREEN, -1);

			// Prep for coordinate mapping
			ColorPoints.push_back(pt);
		}

		// Get 3D World Coordinates
		if (ColorPoints.size()) WorldCoordinates(ColorPoints);

		// Release memory
		std::vector<std::vector<cv::Point>>().swap(contours);
		std::vector<std::vector<cv::Point>>().swap(convexHulls);
		std::vector<Blob>().swap(blobs);
		std::vector<cv::Point>().swap(ColorPoints);
	}

	void WorldCoordinates(std::vector<cv::Point>& cps)
	{
		// Convert depth frame to an appropriate format for deprojection
		rs2::depth_frame d_frame = depth_frame.as<rs2::depth_frame>();

		for (int i = 0; i < cps.size(); i++)
		{
			// Create containers for input and output of deprojection
			float px[2] = { (float)cps[i].x, (float)cps[i].y };
			float pt[3] = {};
			float depth = d_frame.get_distance(cps[i].x, cps[i].y);

			// Deprojection from a pixel on color frame to a point on the 3D coordinates
			rs2_deproject_pixel_to_point(pt, &intrin_depth, px, depth);

			int X_world = (int)(pt[0] * 1000);
			int Y_world = (int)(pt[1] * 1000);
			int Z_world = (int)(pt[2] * 1000);

			if (!(X_world == 0 && Y_world == 0 && Z_world == 0))
			{
				// Display points on depth frame that correspond to those on color frame
				cv::Point tmpDepth = { cps[i].x, cps[i].y };
				cv::circle(depthFrame, tmpDepth, 3, SCALAR_GREEN, -1);

				// Display
				std::string X_str = std::to_string(X_world);
				std::string Y_str = std::to_string(Y_world);
				std::string Z_str = std::to_string(Z_world);
				std::string world_str = std::string(X_str + ", " + Y_str + ", " + Z_str).c_str();
				cv::putText(depthFrame, world_str, tmpDepth, cv::FONT_HERSHEY_COMPLEX, 0.7, SCALAR_RED, 1, CV_AA);
			}
		}
	}
};

int main()
{
	try
	{
		RealSenseApp app;
		app.run();
	}
	catch (std::exception& ex)
	{
		cv::destroyAllWindows();
		std::cout << ex.what() << std::endl;
		return -1;
	}

	return 0;
}