#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <cmath>

void maskIdentical(const cv::Mat & left, const cv::Mat & right,
		cv::Vec3b background,
		cv::Mat & maskedLeft, cv::Mat & maskedRight);

int main(int argc, char** argv) {
	cv::Mat left = cv::imread("/srv/imagediff/test/original.png");
	cv::Mat right = cv::imread("/srv/imagediff/test/moved.png");

	cv::Size leftSize = left.size();
	int smallerDimension = std::min(leftSize.width, leftSize.height);
	int numLevels = (int)(std::log(smallerDimension / 25.) / std::log(2));
	numLevels = std::max(numLevels, 1);

	std::vector<cv::Mat> leftPyramid(numLevels), rightPyramid(numLevels);
	cv::buildPyramid(left, leftPyramid, numLevels - 1);
	cv::buildPyramid(right, rightPyramid, numLevels - 1);

	cv::Vec3b background(255, 255, 255);
	
	for (int level = numLevels - 1; level >= 0; level--) {
		cv::Mat maskedLeft, maskedRight;
		maskIdentical(leftPyramid[level], rightPyramid[level], background,
				maskedLeft, maskedRight);
		cv::imwrite("/tmp/maskedLeft.png", maskedLeft);
		cv::imwrite("/tmp/maskedRight.png", maskedRight);
		break;
	}

	return 0;
}

void maskIdentical(const cv::Mat & left, const cv::Mat & right,
		cv::Vec3b background,
		cv::Mat & maskedLeft, cv::Mat & maskedRight)
{
	cv::Size leftSize = left.size(), rightSize = right.size();
	cv::Size destSize(
			std::max(leftSize.width, rightSize.width),
			std::max(rightSize.height, rightSize.height));
	
	maskedLeft.create(destSize, CV_8UC3);
	maskedRight.create(destSize, CV_8UC3);

	cv::Mat maskRGB = left != right;
	std::vector<cv::Mat> maskParts(3);
	cv::split(maskRGB, maskParts);
	cv::Mat mask = maskParts[0] & maskParts[1] & maskParts[2];

	maskedLeft.setTo(background);
	maskedRight.setTo(background);
	left.copyTo(maskedLeft, mask);
	right.copyTo(maskedRight, mask);
}
