/*
Programmed by Toshikazu Kuroda at Aichi Bunkyo University

<Description>
This is a source code for the 3rd sample program.
It works with RealSense SR300, D415, and D435 camera.
In this program, depth frame is aligned to color frame and then a pixel on color frame is transformed to a point on the 3D coordinates.
There are options for 2D tracking based on color, image subtraction, and background subtraction.
Terminate the program by pressing the 'q' key.

<Additional Note>
Code snippets from align.cpp are embedded.

<References>
For color tracking & Image subtraction, see https://github.com/MicrocontrollersAndMore
For frame alignment, see https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0
https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/examples/sensor-control/api_how_to.h#L277

<System requirements>
1) OS: Windows 10, Mac OS, or Linux Ubuntu 16.04 LTS (also Windows 8.1 if using RealSense SR300)
2) Hardware: USB3.0 port

*/


#include <librealsense2/rs.hpp> 
#include <librealsense2/rsutil.h> 
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

	rs2_error *e = 0;
	float depth_scale = 0.0f;

	// Containers for intrinsic and extrinsic parameters of color and depth cameras
	struct rs2_intrinsics intrin_color;
	struct rs2_intrinsics intrin_depth;
	struct rs2_extrinsics extrin_d2c;
	struct rs2_extrinsics extrin_c2d;

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
		rs2_context* ctx = rs2_create_context(RS2_API_VERSION, &e);
		rs2_device_list* device_list = rs2_query_devices(ctx, &e);
		int dev_count = rs2_get_device_count(device_list, &e);

		if (dev_count)
		{
			rs2_device* dev = rs2_create_device(device_list, 0, &e);
			rs2_sensor_list* sensor_list = rs2_query_sensors(dev, &e);
			int num_of_sensors = rs2_get_sensors_count(sensor_list, &e);

			int is_depth_sensor_found = 0;
			for (int i = 0; i < num_of_sensors; ++i)
			{
				rs2_sensor* sensor = rs2_create_sensor(sensor_list, i, &e);
				is_depth_sensor_found = rs2_is_sensor_extendable_to(sensor, RS2_EXTENSION_DEPTH_SENSOR, &e);
				
				if (is_depth_sensor_found == 1)
				{
					depth_scale = rs2_get_option((const rs2_options*)sensor, RS2_OPTION_DEPTH_UNITS, &e);
					rs2_delete_sensor(sensor);
					break;
				}
				rs2_delete_sensor(sensor);
			}

			std::cout << "depth scale: " << depth_scale << "\n";

			// Enable Color & Depth streams
			cfg.enable_stream(RS2_STREAM_COLOR, COLOR_WIDTH, COLOR_HEIGHT, RS2_FORMAT_BGR8, COLOR_FPS);
			cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_WIDTH, DEPTH_HEIGHT, RS2_FORMAT_Z16, DEPTH_FPS);

			// Start the pipeline for the streams
			profile = pipe.start(cfg);

			std::cout << "\n";

			// Get intrinsic & extrinsic parameters of depth camera
			rs2::video_stream_profile stream_color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
			intrin_color = stream_color.get_intrinsics();
			rs2_distortion model_color = intrin_color.model;
			std::cout << "Intrinsics: Color Camera\n";
			std::cout << "Principal Point         : " << intrin_color.ppx << ", " << intrin_color.ppy << "\n";
			std::cout << "Focal Length            : " << intrin_color.fx << ", " << intrin_color.fy << "\n";
			std::cout << "Distortion Model        : " << model_color << "\n";
			std::cout << "Distortion Coefficients : [" << intrin_color.coeffs[0] << "," << intrin_color.coeffs[1] << "," <<
				intrin_color.coeffs[2] << "," << intrin_color.coeffs[3] << "," << intrin_color.coeffs[4] << "]" << "\n";
			std::cout << "\n";

			rs2::video_stream_profile stream_depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
			intrin_depth = stream_depth.get_intrinsics();
			rs2_distortion model_depth = intrin_depth.model;
			std::cout << "Intrinsics: Depth Camera\n";
			std::cout << "Principal Point         : " << intrin_depth.ppx << ", " << intrin_depth.ppy << "\n";
			std::cout << "Focal Length            : " << intrin_depth.fx << ", " << intrin_depth.fy << "\n";
			std::cout << "Distortion Model        : " << model_depth << "\n";
			std::cout << "Distortion Coefficients : [" << intrin_depth.coeffs[0] << "," << intrin_depth.coeffs[1] << "," <<
				intrin_depth.coeffs[2] << "," << intrin_depth.coeffs[3] << "," << intrin_depth.coeffs[4] << "]" << "\n";
			std::cout << "\n";

			extrin_c2d = stream_color.get_extrinsics_to(stream_depth);
			std::cout << "Extrinsics: Color to Depth\n";
			std::cout << "Translation Vector : [" << extrin_c2d.translation[0] << "," << extrin_c2d.translation[1] << "," << extrin_c2d.translation[2] << "]\n";
			std::cout << "Rotation Matrix    : [" << extrin_c2d.rotation[0] << "," << extrin_c2d.rotation[3] << "," << extrin_c2d.rotation[6] << "]\n";
			std::cout << "                   : [" << extrin_c2d.rotation[1] << "," << extrin_c2d.rotation[4] << "," << extrin_c2d.rotation[7] << "]\n";
			std::cout << "                   : [" << extrin_c2d.rotation[2] << "," << extrin_c2d.rotation[5] << "," << extrin_c2d.rotation[8] << "]\n";
			std::cout << "\n";

			extrin_d2c = stream_depth.get_extrinsics_to(stream_color);
			std::cout << "Extrinsics: Depth to Color\n";
			std::cout << "Translation Vector : [" << extrin_d2c.translation[0] << "," << extrin_d2c.translation[1] << "," << extrin_d2c.translation[2] << "]\n";
			std::cout << "Rotation Matrix    : [" << extrin_d2c.rotation[0] << "," << extrin_d2c.rotation[3] << "," << extrin_d2c.rotation[6] << "]\n";
			std::cout << "                   : [" << extrin_d2c.rotation[1] << "," << extrin_d2c.rotation[4] << "," << extrin_d2c.rotation[7] << "]\n";
			std::cout << "                   : [" << extrin_d2c.rotation[2] << "," << extrin_d2c.rotation[5] << "," << extrin_d2c.rotation[8] << "]\n";
			std::cout << "\n";
		}
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
	template<class GET_DEPTH, class TRANSFER_PIXEL>
	void align_images(const rs2_intrinsics& depth_intrin, const rs2_extrinsics& depth_to_other,
		const rs2_intrinsics& other_intrin, GET_DEPTH get_depth, TRANSFER_PIXEL transfer_pixel)
	{
		// Iterate over the pixels of the depth image
		//#pragma omp parallel for schedule(dynamic)
		for (int depth_y = 0; depth_y < depth_intrin.height; ++depth_y)
		{
			int depth_pixel_index = depth_y * depth_intrin.width;
			for (int depth_x = 0; depth_x < depth_intrin.width; ++depth_x, ++depth_pixel_index)
			{
				// Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
				if (float depth = get_depth(depth_pixel_index))
				{
					// Map the top-left corner of the depth pixel onto the other image
					float depth_pixel[2] = { depth_x - 0.5f, depth_y - 0.5f }, depth_point[3], other_point[3], other_pixel[2];
					rs2_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
					rs2_transform_point_to_point(other_point, &depth_to_other, depth_point);
					rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
					const int other_x0 = static_cast<int>(other_pixel[0] + 0.5f);
					const int other_y0 = static_cast<int>(other_pixel[1] + 0.5f);

					// Map the bottom-right corner of the depth pixel onto the other image
					depth_pixel[0] = depth_x + 0.5f; depth_pixel[1] = depth_y + 0.5f;
					rs2_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
					rs2_transform_point_to_point(other_point, &depth_to_other, depth_point);
					rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
					const int other_x1 = static_cast<int>(other_pixel[0] + 0.5f);
					const int other_y1 = static_cast<int>(other_pixel[1] + 0.5f);

					if (other_x0 < 0 || other_y0 < 0 || other_x1 >= other_intrin.width || other_y1 >= other_intrin.height)
						continue;

					// Transfer between the depth pixels and the pixels inside the rectangle on the other image
					for (int y = other_y0; y <= other_y1; ++y)
					{
						for (int x = other_x0; x <= other_x1; ++x)
						{
							transfer_pixel(depth_pixel_index, y * other_intrin.width + x);
						}
					}
				}
			}
		}
	}

	void align_z_to_other(uint16_t* z_aligned_to_other, const uint16_t* z_pixels, float z_scale, const rs2_intrinsics& z_intrin, const rs2_extrinsics& z_to_other, const rs2_intrinsics& other_intrin)
	{
		auto out_z = (uint16_t *)(z_aligned_to_other);
		align_images(z_intrin, z_to_other, other_intrin,
			[z_pixels, z_scale](int z_pixel_index)
		{
			return z_scale * z_pixels[z_pixel_index];
		},
			[out_z, z_pixels](int z_pixel_index, int other_pixel_index)
		{
			out_z[other_pixel_index] = out_z[other_pixel_index] ?
				std::min((int)out_z[other_pixel_index], (int)z_pixels[z_pixel_index]) :
				z_pixels[z_pixel_index];
		});
	}

	void updateFrame()
	{
		// Wait for next set of frames
		frames = pipe.wait_for_frames();

		// Get color & depth frames and align the depth frame to color frame
		color_frame = frames.get_color_frame();
		depth_frame = frames.get_depth_frame();
		
		// Create a new frame in the rs2::frame format
		rs2::frame_queue q;		
		rs2::processing_block pb([&](rs2::frame f, const rs2::frame_source& src)
		{
			rs2::frame new_frame = src.allocate_video_frame(f.get_profile(), f);
			rs2::video_frame orig = f;

			auto ptr = (uint16_t*)new_frame.get_data();
			memset(ptr, 0, intrin_color.height * intrin_color.width * orig.get_bytes_per_pixel());

			src.frame_ready(new_frame);
		});		
		pb.start(q); // Equivalent to pb >> q;
		pb.invoke(depth_frame); // depth_frame is an input for the pb lambda function
		rs2::frame output_frame = q.wait_for_frame();

		uint16_t* z_aligned_to_other = (uint16_t*)output_frame.get_data();
						
		align_z_to_other(z_aligned_to_other,
			reinterpret_cast<const uint16_t*>(depth_frame.get_data()),
			depth_scale,
			intrin_depth,
			extrin_d2c,
			intrin_color);


		// Convert the default format of the frames to Mat format
		colorFrame = cv::Mat(cv::Size(COLOR_WIDTH, COLOR_HEIGHT), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
		depthFrame = cv::Mat(cv::Size(DEPTH_WIDTH, DEPTH_HEIGHT), CV_16UC1, (void*)output_frame.get_data(), cv::Mat::AUTO_STEP);

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
			rs2_deproject_pixel_to_point(pt, &intrin_color, px, depth);

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