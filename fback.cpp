#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,
                    int mapScale, const cv::Scalar& color);
static void drawColorCodedMap(const cv::Mat& flow, cv::Mat& map);

static cv::Mat prepareImage(cv::Mat input, cv::Size size);


int main(int argc, char** argv)
{
	const int minLevelSize = 32;
	
	if (argc < 4) {
		std::cout << "Usage: fback <input1> <input2> <output>\n";
		return 1;
	}

	cv::Mat leftInput = cv::imread(argv[1]);
	cv::Mat rightInput = cv::imread(argv[2]);

	cv::Size size(
			std::max(leftInput.cols, rightInput.cols),
			std::max(leftInput.rows, rightInput.rows));

	const int numLevels = 1;

	std::cout << "Creating " << size.width << "x" << size.height << " flow field\n";
	
	cv::Mat left = prepareImage(leftInput, size);
	cv::Mat right = prepareImage(rightInput, size);
	cv::Mat flow;

	if (false) {
		cv::calcOpticalFlowFarneback(left, right, flow, 0.5, numLevels, 15, 3, 5, 1.2, 0);
	} else {
		cv::calcOpticalFlowSF(left, right, flow, numLevels, 4, 2);
	}
	
	cv::Mat visual;
	if (false) {
		const int mapScale = 2;
		cv::Mat leftBlended(size, CV_8UC3);
		leftBlended = cv::Scalar(128, 128, 128);
		leftBlended(cv::Rect(cv::Point(0, 0), leftInput.size())) += leftInput / 2;
		cv::resize(leftBlended, visual, size * mapScale, 0, 0, cv::INTER_NEAREST);
		std::cout << "Drawing map " << visual.cols << "x" << visual.rows << "\n";
		drawOptFlowMap(flow, visual, 8, mapScale, cv::Scalar(0, 255, 0));
	} else {
		drawColorCodedMap(flow, visual);
	}

	cv::imwrite(argv[3], visual);
	return 0;
}

static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,
		int mapScale, const cv::Scalar& color)
{
	for(int y = 0; y < flow.rows; y += step) {
		for(int x = 0; x < flow.cols; x += step) {
			const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
			cv::line(cflowmap, mapScale * cv::Point(x,y),
					mapScale * cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
					color);
			cv::circle(cflowmap, mapScale * cv::Point(x,y), 2, color, -1);
		}
	}
}

static void drawColorCodedMap(const cv::Mat& flow, cv::Mat& map) {
	// Calculate maximum
	float maxVal = 0;
	for (int y = 0; y < flow.rows; y++) {
		for (int x = 0; x < flow.cols; x++) {
			const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
			maxVal = std::max(maxVal ,std::sqrt(fxy.dot(fxy)));
		}
	}

	cv::Mat_<cv::Vec3b> hsvMap(flow.size());
	// Create map
	for (int y = 0; y < flow.rows; y++) {
		for (int x = 0; x < flow.cols; x++) {
			const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
			cv::Vec3b & hsv = hsvMap(y, x);
			hsv[0] = cv::saturate_cast<unsigned char>(cv::fastAtan2(fxy.y, fxy.x) / 2);
			hsv[1] = 255;
			hsv[2] = cv::saturate_cast<unsigned char>(std::sqrt(fxy.dot(fxy)) / maxVal * 255.);
		}
	}
	// Overlay legend
	const int radius = maxVal;
	const int centre = radius + 10;
	for (int y = centre - radius; y <= centre + radius; y++) {
		int dx = cvRound(std::sqrt(std::fabs(radius*radius - (y - centre) * (y - centre))));
		for (int x = centre - dx; x <= centre + dx; x++) {
			cv::Vec3b & hsv = hsvMap(y, x);
			float angle = cv::fastAtan2(y - centre, x - centre);
			double r = std::sqrt((y - centre) * (y - centre) + (x - centre) * (x - centre));
			hsv[0] = cv::saturate_cast<unsigned char>(angle / 2);
			hsv[1] = 255;
			hsv[2] = cv::saturate_cast<unsigned char>(r * 255. / radius);
		}
	}

	map.create(flow.size(), CV_8UC3);
	cv::cvtColor(hsvMap, map, CV_HSV2BGR);
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

