#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,
                    double, const cv::Scalar& color);

static cv::Mat prepareImage(cv::Mat input, cv::Size size);


int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cout << "Usage: fback <input1> <input2> <output>\n";
		return 1;
	}

	cv::Mat leftInput = cv::imread(argv[1]);
	cv::Mat rightInput = cv::imread(argv[2]);

	cv::Size size(
			std::max(leftInput.cols, rightInput.cols),
			std::max(leftInput.rows, rightInput.rows));

	std::cout << "Creating " << size.width << "x" << size.height << " flow field\n";
	
	cv::Mat left = prepareImage(leftInput, size);
	cv::Mat right = prepareImage(rightInput, size);
	cv::Mat flow, visual(size, CV_8UC3);

	cv::calcOpticalFlowFarneback(left, right, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	visual = cv::Scalar(128, 128, 128);
	visual(cv::Rect(cv::Point(0, 0), leftInput.size())) += leftInput / 2;

	drawOptFlowMap(flow, visual, 32, 1.5, cv::Scalar(0, 255, 0));

	cv::imwrite(argv[3], visual);
	return 0;
}

static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,
		double, const cv::Scalar& color)
{
	for(int y = 0; y < cflowmap.rows; y += step) {
		for(int x = 0; x < cflowmap.cols; x += step) {
			const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
			cv::line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
					color);
			cv::circle(cflowmap, cv::Point(x,y), 2, color, -1);
		}
	}
}

static cv::Mat prepareImage(cv::Mat input, cv::Size size) {
	// Convert colour
	cv::Mat gray;
	cv::cvtColor(input, gray, CV_BGR2GRAY);
	// Background fill
	cv::Mat dest = cv::Mat::ones(size, CV_8UC1) * 255;

	// Expand to desired dimensions by repeating the edge row/column
	gray.copyTo(dest(cv::Rect(cv::Point(0, 0), gray.size())));
	cv::Mat edge = gray.colRange(gray.cols - 1, gray.cols);
	for (int x = gray.cols; x < size.width; x++) {
		dest(cv::Rect(x, 0, 1, edge.rows)) = dest;
	}

	edge = gray.rowRange(gray.rows - 1, gray.rows);
	for (int y = gray.rows; y < size.height; y++) {
		dest(cv::Rect(0, y, edge.cols, 1)) = edge;
	}

	return dest;
}

