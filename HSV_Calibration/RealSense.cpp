/*
Programmed by Toshikazu Kuroda at Aichi Bunkyo University

<Description>
This is a source code for calibrating HSV parameters.

1) Press 'a' to initiate the calibration.
2) Press 'l' or 'u' to select the lower and upper range, respectively.
3) Press 'h', 's', or 'v' to select Hue, Saturation, and Value, respectively.
4) Press an arrow key to change the parameter.

It works with RealSense SR300, D415, and D435 camera.
Terminate the program by pressing the 'q' key.

<Important Note>
This program can be built only on Windows OS given the use of #include <conio.h> for using a keyboard.
For a different type of OS, replace #include <conio.h> with something equivalent and make appropriate changes in the source code.

<Reference>
For color tracking, see https://github.com/MicrocontrollersAndMore

<System requirements>
1) OS: Windows 10 (also Windows 8.1 if using RealSense SR300)
2) Hardware: USB3.0 port

*/


#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <conio.h>

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

	// Stream parameters
	const int COLOR_WIDTH = 640;
	const int COLOR_HEIGHT = 480;
	const int COLOR_FPS = 60;
	const int DEPTH_WIDTH = 640;
	const int DEPTH_HEIGHT = 480;
	const int DEPTH_FPS = 60;

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

	// HSV Calibration
	int HSV_lower_or_higher = 0;
	int HSV_flag = 0;

	struct HSV_LOWER
	{
		int Hue_Low = 0;
		int Sat_Low = 155;
		int Val_Low = 155;
		int Hue_High = 20;		
		int Sat_High = 255;		
		int Val_High = 255;
	};

	struct HSV_UPPER
	{
		int Hue_Low = 165;
		int Sat_Low = 155;
		int Val_Low = 155;
		int Hue_High = 180;
		int Sat_High = 255;
		int Val_High = 255;
	};

	// Memory
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> convexHulls;
	std::vector<Blob> blobs;

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

		// Release memory
		std::vector<std::vector<cv::Point>>().swap(contours);
		std::vector<std::vector<cv::Point>>().swap(convexHulls);
		std::vector<Blob>().swap(blobs);

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
		cv::Mat hsvFrame, threshLow, threshHigh, resultFrame_hsv, resultFrame;
		struct HSV_LOWER HSV_lower;
		struct HSV_UPPER HSV_upper;

		start = std::chrono::system_clock::now();

		while (1)
		{
			updateFrame();
			colorFrame_copy = colorFrame.clone();
			FPS();

			if (_kbhit())
			{
				switch (_getch())
				{
				case 'a':
					std::cout << "HSV Lower range\n";
					std::cout << "  Hue: " << HSV_lower.Hue_Low << "-" << HSV_lower.Hue_High << "\n";
					std::cout << "  Sat: " << HSV_lower.Sat_Low << "-" << HSV_lower.Sat_High << "\n";
					std::cout << "  Val: " << HSV_lower.Val_Low << "-" << HSV_lower.Val_High << "\n";
					std::cout << "HSV Higher range\n";
					std::cout << "  Hue: " << HSV_upper.Hue_Low << "-" << HSV_upper.Hue_High << "\n";
					std::cout << "  Sat: " << HSV_upper.Sat_Low << "-" << HSV_upper.Sat_High << "\n";
					std::cout << "  Val: " << HSV_upper.Val_Low << "-" << HSV_upper.Val_High << "\n";
					std::cout << "\n";
					std::cout << "Press 'l' or 'u' for the lower and upper range, respectively\n";
					std::cout << "\n";
					HSV_lower_or_higher = 0;
					HSV_flag = 0;
					break;
				case 'l':
					std::cout << "Lower range in effect. Press 'h', 's', or 'v' for Hue, Saturation, and Value, respectively\n";
					std::cout << "\n";
					HSV_lower_or_higher = 1;
					HSV_flag = 0;
					break;
				case 'u':
					std::cout << "Upper range in effect. Press 'h', 's', or 'v' for Hue, Saturation, and Value, respectively\n";
					std::cout << "\n";
					HSV_lower_or_higher = 2;
					HSV_flag = 0;
					break;
				case 'h':
					switch (HSV_lower_or_higher)
					{
					case 1:
						std::cout << "Hue (lower range) in effect\n";
						std::cout << "Press an arrow key\n";
						HSV_flag = 1;						
						break;
					case 2:
						std::cout << "Hue (upper range) in effect\n";
						std::cout << "Press an arrow key\n";
						HSV_flag = 4;						
						break;
					}					
					break;
				case 's':
					switch (HSV_lower_or_higher)
					{
					case 1:
						std::cout << "Saturation (lower range) in effect\n";
						std::cout << "Press an arrow key\n";
						HSV_flag = 2;
						break;
					case 2:
						std::cout << "Saturation (upper range) in effect\n";
						std::cout << "Press an arrow key\n";
						HSV_flag = 5;
						break;
					}					
					break;
				case 'v':
					switch (HSV_lower_or_higher)
					{
					case 1:
						std::cout << "Value (lower range) in effect\n";
						std::cout << "Press an arrow key\n";
						HSV_flag = 3;
						break;
					case 2:
						std::cout << "Value (upper range) in effect\n";
						std::cout << "Press an arrow key\n";
						HSV_flag = 6;
						break;
					}					
					break;


				case 0xe0: // Arrow keys
					switch (_getch())
					{
					case 0x48: // Up for Low
						switch (HSV_flag)
						{
						case 1: // Hue (Lower)
							if (HSV_lower.Hue_Low != 180)
							{
								HSV_lower.Hue_Low = HSV_lower.Hue_Low + 5;
								std::cout << HSV_lower.Hue_Low << "\n";
							}
							break;
						case 2: // Sat (Lower)
							if (HSV_lower.Sat_Low != 255)
							{
								HSV_lower.Sat_Low = HSV_lower.Sat_Low + 5;
								std::cout << HSV_lower.Sat_Low << "\n";
							}
							break;
						case 3: // Val (Lower)
							if (HSV_lower.Val_Low != 255)
							{
								HSV_lower.Val_Low = HSV_lower.Val_Low + 5;
								std::cout << HSV_lower.Val_Low << "\n";
							}
							break;
						case 4: // Hue (upper)
							if (HSV_upper.Hue_Low != 180)
							{
								HSV_upper.Hue_Low = HSV_upper.Hue_Low + 5;
								std::cout << HSV_upper.Hue_Low << "\n";
							}
							break;
						case 5: // Sat (upper)
							if (HSV_upper.Sat_Low != 255)
							{
								HSV_upper.Sat_Low = HSV_upper.Sat_Low + 5;
								std::cout << HSV_upper.Sat_Low << "\n";
							}
							break;
						case 6: // Val (upper)
							if (HSV_upper.Val_Low != 255)
							{
								HSV_upper.Val_Low = HSV_upper.Val_Low + 5;
								std::cout << HSV_upper.Val_Low << "\n";
							}
							break;
						}
						break;
					case 0x50: // Down for Low
						switch (HSV_flag)
						{
						case 1: // Hue (Lower)
							if (HSV_lower.Hue_Low != 0)
							{
								HSV_lower.Hue_Low = HSV_lower.Hue_Low - 5;
								std::cout << HSV_lower.Hue_Low << "\n";
							}
							break;
						case 2: // Sat (Lower)
							if (HSV_lower.Sat_Low != 0)
							{
								HSV_lower.Sat_Low = HSV_lower.Sat_Low - 5;
								std::cout << HSV_lower.Sat_Low << "\n";
							}
							break;
						case 3: // Val (Lower)
							if (HSV_lower.Val_Low != 0)
							{
								HSV_lower.Val_Low = HSV_lower.Val_Low - 5;
								std::cout << HSV_lower.Val_Low << "\n";
							}
							break;
						case 4: // Hue (upper)
							if (HSV_upper.Hue_Low != 0)
							{
								HSV_upper.Hue_Low = HSV_upper.Hue_Low - 5;
								std::cout << HSV_upper.Hue_Low << "\n";
							}
							break;
						case 5: // Sat (upper)
							if (HSV_upper.Sat_Low != 0)
							{
								HSV_upper.Sat_Low = HSV_upper.Sat_Low - 5;
								std::cout << HSV_upper.Sat_Low << "\n";
							}
							break;
						case 6: // Val (upper)
							if (HSV_upper.Val_Low != 0)
							{
								HSV_upper.Val_Low = HSV_upper.Val_Low - 5;
								std::cout << HSV_upper.Val_Low << "\n";
							}
							break;
						}
						break;
					case 0x4d: // Up for High
						switch (HSV_flag)
						{
						case 1: // Hue (Lower)
							if (HSV_lower.Hue_High != 180)
							{
								HSV_lower.Hue_High = HSV_lower.Hue_High + 5;
								std::cout << HSV_lower.Hue_High << "\n";
							}
							break;
						case 2: // Sat (Lower)
							if (HSV_lower.Sat_High != 255)
							{
								HSV_lower.Sat_High = HSV_lower.Sat_High + 5;
								std::cout << HSV_lower.Sat_High << "\n";
							}
							break;
						case 3: // Val (Lower)
							if (HSV_lower.Val_High != 255)
							{
								HSV_lower.Val_High = HSV_lower.Val_High + 5;
								std::cout << HSV_lower.Val_High << "\n";
							}
							break;
						case 4: // Hue (upper)
							if (HSV_upper.Hue_High != 180)
							{
								HSV_upper.Hue_High = HSV_upper.Hue_High + 5;
								std::cout << HSV_upper.Hue_High << "\n";
							}
							break;
						case 5: // Sat (upper)
							if (HSV_upper.Sat_High != 255)
							{
								HSV_upper.Sat_High = HSV_upper.Sat_High + 5;
								std::cout << HSV_upper.Sat_High << "\n";
							}
							break;
						case 6: // Val (upper)
							if (HSV_upper.Val_High != 255)
							{
								HSV_upper.Val_High = HSV_upper.Val_High + 5;
								std::cout << HSV_upper.Val_High << "\n";
							}
							break;
						}
						break;
					case 0x4b: // Down for High
						switch (HSV_flag)
						{
						case 1: // Hue (Lower)
							if (HSV_lower.Hue_High != 0)
							{
								HSV_lower.Hue_High = HSV_lower.Hue_High - 5;
								std::cout << HSV_lower.Hue_High << "\n";
							}
							break;
						case 2: // Sat (Lower)
							if (HSV_lower.Sat_High != 0)
							{
								HSV_lower.Sat_High = HSV_lower.Sat_High - 5;
								std::cout << HSV_lower.Sat_High << "\n";
							}
							break;
						case 3: // Val (Lower)
							if (HSV_lower.Val_High != 0)
							{
								HSV_lower.Val_High = HSV_lower.Val_High - 5;
								std::cout << HSV_lower.Val_High << "\n";
							}
							break;
						case 4: // Hue (upper)
							if (HSV_upper.Hue_High != 0)
							{
								HSV_upper.Hue_High = HSV_upper.Hue_High - 5;
								std::cout << HSV_upper.Hue_High << "\n";
							}
							break;
						case 5: // Sat (upper)
							if (HSV_upper.Sat_High != 0)
							{
								HSV_upper.Sat_High = HSV_upper.Sat_High - 5;
								std::cout << HSV_upper.Sat_High << "\n";
							}
							break;
						case 6: // Val (upper)
							if (HSV_upper.Val_High != 0)
							{
								HSV_upper.Val_High = HSV_upper.Val_High - 5;
								std::cout << HSV_upper.Val_High << "\n";
							}
							break;
						}
						break;
					}
					break;
				}
			}

			
			// Color tracking
			cv::cvtColor(colorFrame, hsvFrame, CV_BGR2HSV);
			cv::inRange(hsvFrame, cv::Scalar(HSV_lower.Hue_Low, HSV_lower.Sat_Low, HSV_lower.Val_Low), cv::Scalar(HSV_lower.Hue_High, HSV_lower.Sat_High, HSV_lower.Val_High), threshLow);
			cv::inRange(hsvFrame, cv::Scalar(HSV_upper.Hue_Low, HSV_upper.Sat_Low, HSV_upper.Val_Low), cv::Scalar(HSV_upper.Hue_High, HSV_upper.Sat_High, HSV_upper.Val_High), threshHigh);
			cv::add(threshLow, threshHigh, resultFrame_hsv);
			cv::imshow("Color Tracking", resultFrame_hsv);

			// Optional edits for reducing noise
			cv::GaussianBlur(resultFrame_hsv, resultFrame_hsv, cv::Size(5, 5), 0);
			cv::dilate(resultFrame_hsv, resultFrame_hsv, structuringElement3x3);
			cv::erode(resultFrame_hsv, resultFrame_hsv, structuringElement3x3);
			
			resultFrame = resultFrame_hsv.clone();						

			DetectCenterOfObject(resultFrame);

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

	void DetectCenterOfObject(cv::Mat& result)
	{
		// Convert Mat to a vector of points			
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
		}

		// Release memory
		std::vector<std::vector<cv::Point>>().swap(contours);
		std::vector<std::vector<cv::Point>>().swap(convexHulls);
		std::vector<Blob>().swap(blobs);
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