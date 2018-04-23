/*
Programmed by Toshikazu Kuroda at Aichi Bunkyo University

<Description>
This is a source code for the 1st sample program.
It works with RealSense SR300, D415, and D435 camera, showing streams of color and depth frames using RealSense SDK 2.0 & OpenCV.
Terminate the program by pressing the 'q' key.

<System requirements>
1) OS: Windows 10, Mac OS, or Linux Ubuntu 16.04 LTS (also Windows 8.1 if using RealSense SR300)
2) Hardware: USB3.0 port

*/


#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <chrono>

class RealSenseApp
{
private:
	// Classes
	rs2::pipeline pipe;
	rs2::config cfg;
	rs2::frameset frames;
	rs2::frame color_frame;
	rs2::frame depth_frame;

	// Stream parameters
	const int COLOR_WIDTH = 640;
	const int COLOR_HEIGHT = 480;
	const int COLOR_FPS = 60;
	const int DEPTH_WIDTH = 640;
	const int DEPTH_HEIGHT = 480;
	const int DEPTH_FPS = 60;

	// Mat data containers
	cv::Mat colorFrame;
	cv::Mat depthFrame;

	// Set color parameters
	const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
	const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
	const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
	const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
	const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

	// FPS
	std::chrono::system_clock::time_point start, end;
	int fps = 0;

public:
	// Constructor
	RealSenseApp()
	{
		// Enable Color & Depth streams
		cfg.enable_stream(RS2_STREAM_COLOR, COLOR_WIDTH, COLOR_HEIGHT, RS2_FORMAT_BGR8, COLOR_FPS);
		cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_Z16, DEPTH_FPS);

		// Start the pipeline for the streams
		pipe.start(cfg);
	}

	// Destructor
	~RealSenseApp()
	{
		// Terminate the pipeline
		pipe.stop();

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

		start = std::chrono::system_clock::now();

		while (1)
		{
			updateFrame();
			FPS();
			
			// Show frame streams
			if (colorFrame.rows != 0 && colorFrame.cols != 0) cv::imshow("Color Image", colorFrame);
			if (depthFrame.rows != 0 && depthFrame.cols != 0) cv::imshow("Depth Image", depthFrame);

			// Press the "q" key to terminate the program
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
		color_frame = frames.get_color_frame();
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