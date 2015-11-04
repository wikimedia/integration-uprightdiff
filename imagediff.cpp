#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdlib>
#include <iostream>
#include <cmath>

#include "BlockMotionSearch.h"

typedef cv::Mat_<cv::Vec3b> Mat3b;
typedef cv::Mat_<int> Mat1i;

static Mat3b convertInput(const char * label, cv::Mat & input, const cv::Size & size);
static int expandRight(const Mat3b & m1, const Mat3b & m2, const cv::Rect sourceRect, int dy);
static void paintSubBlockLine(Mat1i & motion, const Mat3b & m1, const Mat3b & m2,
		const cv::Point & start, const cv::Point & step);
static unsigned char bgrToGrey(const cv::Vec3b & bgr);
static Mat1i scaleUpMotion(Mat1i & blockMotion, int blockSize, const cv::Size & destSize);
static void annotateMotion(const Mat1i & motion, Mat3b & visual);
static cv::Point findMaskCentre(const cv::Mat_<unsigned char> mask, int value, int totalArea);
void arrowedLine(Mat3b img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color,
           int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1);

static const int NOT_FOUND = 0x7fffffff;

int main(int argc, char** argv) {
	if (argc < 4) {
		std::cout << "Usage: imagediff <input1> <input2> <output>\n";
		return 1;
	}

	cv::Mat aliceInput = cv::imread(argv[1]);
	cv::Mat bobInput = cv::imread(argv[2]);

	cv::Size sharedSize(
			std::max(aliceInput.cols, bobInput.cols),
			std::max(aliceInput.rows, bobInput.rows));

	Mat3b alice = convertInput("first", aliceInput, sharedSize);
	Mat3b bob = convertInput("second", bobInput, sharedSize);

	aliceInput.release();
	bobInput.release();

	const int blockSize = 16;
	const int windowSize = 200;

	// Calculate block motion by exhaustive search
	std::cout << "Searching for motion...\n";
	Mat1i blockMotion = BlockMotionSearch::Search(bob, alice, blockSize, windowSize);
	

	// Scale up block motion matrix
	Mat1i motion = scaleUpMotion(blockMotion, blockSize, sharedSize);
	cv::imwrite("/tmp/imagediff/prepaint.png", motion * 10 + 128);

	std::cout << "Expanding motion blocks\n";

	// Expand block motion into sub-block NOT_FOUND regions
	for (int y = 0; y < bob.rows; y++) {
		// Paint right
		paintSubBlockLine(motion, bob, alice, cv::Point(0, y), cv::Point(1, 0));
		// Paint left
		paintSubBlockLine(motion, bob, alice, cv::Point(bob.cols - 1, y), cv::Point(-1, 0));
	}
	for (int x = 0; x < bob.cols; x++) {
		// Paint down
		paintSubBlockLine(motion, bob, alice, cv::Point(x, 0), cv::Point(0, 1));
		// Paint up
		paintSubBlockLine(motion, bob, alice, cv::Point(x, bob.rows - 1), cv::Point(0, -1));
	}
	cv::imwrite("/tmp/imagediff/postpaint.png", motion * 10 + 128);

	std::cout << "Generating visualization\n";

	// Prepare moved image
	Mat3b moved(bob.rows, bob.cols, CV_8UC3);
	moved.setTo(cv::Vec3b(255, 0, 255));
	for (int y = 0; y < bob.rows; y++) {
		for (int x = 0; x < bob.cols; x++) {
			int dy = motion(y, x);
			if (dy != NOT_FOUND) {
				if (y + dy >= moved.rows || y + dy < 0) {
					std::cout << "Error: out of bounds: (" << x << ", " << y << " + " << dy << ")\n";
					return 1;
				}
				moved(y, x) = alice(y + dy, x);
			}
		}
	}

	// Compute visualisation and motion residuals
	Mat3b visual(sharedSize, CV_8UC3);
	visual.setTo(cv::Vec3b(128, 128, 128));
	int residualArea = 0;
	for (int y = 0; y < bob.rows; y++) {
		for (int x = 0; x < bob.cols; x++) {
			if (moved(y, x) == bob(y, x)) {
				unsigned char level = 127 + bgrToGrey(moved(y, x)) / 2;
				visual(y, x) = cv::Vec3b(level, level, level);
			} else if (motion(y, x) == NOT_FOUND) {
				cv::Vec3b ac = alice(y, x);
				cv::Vec3b bc = bob(y, x);
				if (ac == bc) {
					visual(y, x) = ac;
				} else {
					visual(y, x) = cv::Vec3b(0, bgrToGrey(bc), bgrToGrey(ac));
					residualArea ++;
				}
			} else {
				cv::Vec3b mc = moved(y, x);
				cv::Vec3b bc = bob(y, x);
				visual(y, x) = cv::Vec3b(0, bgrToGrey(bc), bgrToGrey(mc));
				residualArea ++;
			}
		}
	}

	// Draw motion annotations
	annotateMotion(motion, visual);

	cv::imwrite("/tmp/imagediff/alice.png", alice);
	cv::imwrite("/tmp/imagediff/bob.png", bob);
	cv::imwrite("/tmp/imagediff/moved.png", moved);
	cv::imwrite("/tmp/imagediff/mask.png", alice == bob);
	cv::imwrite(argv[3], visual);

	std::cout << "Residual: " << residualArea << " pixels\n";
}

static unsigned char bgrToGrey(const cv::Vec3b & bgr) {
	return cv::saturate_cast<unsigned char>(
			76 * bgr[2] / 255     // Blue
			+ 150 * bgr[1] / 255  // Green
			+ 29 * bgr[0] / 255); // Red
}

static Mat3b convertInput(const char * label, cv::Mat & input, const cv::Size & size) {
	if (input.type() != CV_8UC3) {
		std::cout << "The " << label << " image is invalid or has the wrong pixel type\n";
		std::exit(1);
	}
	Mat3b ret(size, CV_8UC3);
	ret.setTo(cv::Vec3b(128, 128, 128));
	input.copyTo(ret(cv::Rect(cv::Point(), input.size())));
	return ret;
}

static Mat1i scaleUpMotion(Mat1i & blockMotion, int blockSize, const cv::Size & destSize) {
	Mat1i motion(destSize, CV_32SC1);
	Mat1i notFound(1, 1, CV_32SC1);
	notFound(0, 0) = NOT_FOUND;
	int x, y, xIndex, yIndex;
	for (y = 0, yIndex = 0; yIndex < blockMotion.rows; yIndex++, y += blockSize) {
		for (x = 0, xIndex = 0; xIndex < blockMotion.cols; xIndex++, x += blockSize) {
			cv::Rect sourceRect(xIndex, yIndex, 1, 1);
			cv::Rect destRect(xIndex * blockSize, yIndex * blockSize, blockSize, blockSize);
			cv::repeat(blockMotion(sourceRect), blockSize, blockSize, motion(destRect));
		}
		cv::Rect edgeRect(x, y, destSize.width - x, blockSize);
		if (edgeRect.width > 0) {
			cv::repeat(notFound, edgeRect.height, edgeRect.width, motion(edgeRect));
		}
	}

	{
		cv::Rect edgeRect(0, y, destSize.width, destSize.height - y);
		if (edgeRect.height > 0) {
			cv::repeat(notFound, edgeRect.height, edgeRect.width, motion(edgeRect));
		}
	}
	return motion;
}

static void paintSubBlockLine(Mat1i & motion, const Mat3b & m1, const Mat3b & m2,
		const cv::Point & start, const cv::Point & step)
{
	cv::Point pos = start;
	cv::Rect bounds(cv::Point(), motion.size());
	int prevMotion = NOT_FOUND;
	while (bounds.contains(pos)) {
		int curMotion = motion.at<int>(pos);
		if (curMotion == NOT_FOUND && prevMotion != NOT_FOUND) {
			cv::Point destPos = pos + cv::Point(0, prevMotion);
			if (bounds.contains(destPos) && m1(pos) == m2(destPos)) {
				motion.at<int>(pos) = prevMotion;
			}
		}
		prevMotion = motion.at<int>(pos);
		pos += step;
	}
}

static void annotateMotion(const Mat1i & motionInput, Mat3b & visual) {
	Mat3b contourVis = cv::Mat::zeros(visual.size(), CV_8UC3);

	std::vector<cv::Scalar> palette;
	palette.push_back(cv::Scalar(0xff, 0x00, 0x00));
	palette.push_back(cv::Scalar(0xff, 0x80, 0x00));
	palette.push_back(cv::Scalar(0xff, 0x00, 0x80));
	int paletteIndex = 0;

	// Find motion regions by flood filling
	Mat1i motion(cv::Size(motionInput.cols + 2, motionInput.rows + 2), CV_32SC1);
	motion = NOT_FOUND;
	motionInput.copyTo(motion(cv::Rect(cv::Point(1, 1), motionInput.size())));
	int regionIndex = 2;
	cv::Mat_<unsigned char> mask(cv::Size(motion.cols + 2, motion.rows + 2), CV_8UC1);
	mask = 0;
	const int minArea = 50;
	const int maxContourIndex = 255;
	for (int y = 1; y < motion.rows - 1 && regionIndex <= maxContourIndex; y++) {
		for (int x = 1; x < motion.cols - 1 && regionIndex <= maxContourIndex; x++) {
			if (mask(y + 1, x + 1)) {
				continue;
			}
			int currentMotion = motion(y, x);
			if (currentMotion == 0 || currentMotion == NOT_FOUND) {
				continue;
			}
			int area = floodFill(
					motion,
					mask,
					cv::Point(x, y), // seedPoint
					cv::Scalar(), // newVal
					nullptr, // rect
					cv::Scalar(), // loDiff
					cv::Scalar(), // upDiff
					4  // connectivity
					| (regionIndex << 8) // mask value
					| cv::FLOODFILL_MASK_ONLY
					);
			std::cout << "Found region with area " << area << " at (" << x << ", " << y << "), "
				"motion = " << currentMotion << "\n";
			if (area < minArea) {
				// Too small for contour, fill instead
				cv::Mat_<unsigned char> currentMask = (mask == regionIndex);
				mask.setTo(1, currentMask);
				contourVis.setTo(palette[paletteIndex],
					currentMask(cv::Rect(cv::Point(2, 2), mask.size() - cv::Size(4, 4))));
				paletteIndex = (paletteIndex + 1) % palette.size();
			} else {
				// Draw arrow
				cv::Point centrePoint = findMaskCentre(mask, regionIndex, area);
				cv::Scalar colour = palette[regionIndex % palette.size()];
				arrowedLine(contourVis, centrePoint + cv::Point(0, currentMotion),
						centrePoint, colour);

				// Draw arrow label
				std::string text = std::to_string(std::abs(currentMotion));
				cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN,
						1, 1, nullptr);
				cv::putText(contourVis, text,
						centrePoint + cv::Point(2, currentMotion / 2 + textSize.height / 2),
						cv::FONT_HERSHEY_PLAIN,	1, colour);
				std::cout << "(" << centrePoint.x << ", " << centrePoint.y << ") " << text << "\n";

				// Find contours


				regionIndex++;
			}
		}
	}

	// Find contours of each area and draw them
	for (int i = 2; i < regionIndex; i++) {
		std::vector<std::vector<cv::Point>> contours;
		findContours(cv::Mat(mask == i), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		drawContours(contourVis, contours, -1, palette[i % palette.size()],
				1, 8, cv::noArray(), INT_MAX, cv::Point(-2, -2));
	}

	// Blend with destination
	for (int y = 0; y < visual.rows; y++) {
		for (int x = 0; x < visual.cols; x++) {
			if (contourVis(y, x) != cv::Vec3b()) {
				visual(y, x) = visual(y, x) / 2 + contourVis(y, x) / 2;
			}
		}
	}
}

static cv::Point findMaskCentre(const cv::Mat_<unsigned char> mask, int value, int totalArea) {
	int64_t sumX = 0, sumY = 0;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask(y, x) == value) {
				sumX += x;
				sumY += y;
			}
		}
	}
	return cv::Point(int(sumX / totalArea), int(sumY / totalArea));
}

/**
 * This is a copy of cv::arrowedLine(), which wasn't available in the version of
 * OpenCV I was linking to.
 */
void arrowedLine(Mat3b img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color,
           int thickness, int line_type, int shift, double tipLength)
{
    const double tipSize = cv::norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow

	cv::line(img, pt1, pt2, color, thickness, line_type, shift);

    const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );

	cv::Point p(cvRound(pt2.x + tipSize * std::cos(angle + CV_PI / 4)),
        cvRound(pt2.y + tipSize * std::sin(angle + CV_PI / 4)));
	cv::line(img, p, pt2, color, thickness, line_type, shift);

    p.x = cvRound(pt2.x + tipSize * std::cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * std::sin(angle - CV_PI / 4));
	cv::line(img, p, pt2, color, thickness, line_type, shift);
}

