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
	const int windowSize = 100;

	// Calculate block motion by exhaustive search
	std::cout << "Searching for motion...\n";
	Mat1i blockMotion = BlockMotionSearch::Search(alice, bob, blockSize, windowSize);
	
/*
	// Merge blocks by flood filling
	int regionIndex = 1;
	cv::Mat seedPoints(255, 1, CV_32SC2);
	cv::Mat mask(yBlockCount + 2, xBlockCount + 2, CV_8U1);
	for (int yIndex = 0; yIndex < yBlockCount && regionIndex <= 255; yIndex++) {
		for (int xIndex = 0; xIndex < xBlockCount && regionIndex <= 255; xIndex++) {
			if (mask(yIndex, xIndex)) {
				continue;
			}
			floodFill(
					blockMotion,
					mask,
					cv::Point(yIndex, xIndex), // seedPoint
					cv::Scalar(), // newVal
					nullptr, // rect
					cv::Scalar(), // loDiff
					cv::Scalar(), // upDiff
					4  // connectivity
					| (regionIndex << 8) // mask value
					| FLOODFILL_MASK_ONLY
					);
			seedPoints(regionIndex++) = cv::Vec2i(xIndex, yIndex);
		}
	}

	// Crop the mask boundaries, which are no longer needed
	mask = mask(cv::Rect(1, 1, mask.cols - 2, mask.rows - 2));
*/

	// Scale up block motion matrix
	Mat1i motion = scaleUpMotion(blockMotion, blockSize, sharedSize);

	std::cout << "Expanding motion blocks\n";
	cv::imwrite("/tmp/imagediff/prepaint.png", motion * 10 + 128);

	// Expand block motion into sub-block NOT_FOUND regions
	for (int y = 0; y < alice.rows; y++) {
		// Paint right
		paintSubBlockLine(motion, alice, bob, cv::Point(0, y), cv::Point(1, 0));
		// Paint left
		paintSubBlockLine(motion, alice, bob, cv::Point(alice.cols - 1, y), cv::Point(-1, 0));
	}
	for (int x = 0; x < alice.cols; x++) {
		// Paint down
		paintSubBlockLine(motion, alice, bob, cv::Point(x, 0), cv::Point(0, 1));
		// Paint up
		paintSubBlockLine(motion, alice, bob, cv::Point(x, alice.rows - 1), cv::Point(0, -1));
	}
	cv::imwrite("/tmp/imagediff/postpaint.png", motion * 10 + 128);

	std::cout << "Generating visualization\n";

	// Prepare moved image
	Mat3b moved(bob.rows, bob.cols, CV_8UC3);
	moved.setTo(cv::Vec3b(0, 0, 0));
	for (int y = 0; y < alice.rows; y++) {
		for (int x = 0; x < alice.cols; x++) {
			int dy = motion(y, x);
			if (dy != NOT_FOUND) {
				if (y + dy >= moved.rows || y + dy < 0) {
					std::cout << "Error: out of bounds: (" << x << ", " << y << " + " << dy << ")\n";
					return 1;
				}
				moved(y + dy, x) = alice(y, x);
			}
		}
	}

	// Compute visualisation and motion residuals
	Mat3b visual(sharedSize, CV_8UC3);
	visual.setTo(cv::Vec3b(128, 128, 128));
	int residualArea = 0;
	for (int y = 0; y < alice.rows; y++) {
		for (int x = 0; x < alice.cols; x++) {
			if (moved(y, x) == bob(y, x)) {
				unsigned char level = 127 + bgrToGrey(moved(y, x)) / 2;
				visual(y, x) = cv::Vec3b(level, level, level);
			} else if (motion(y, x) == NOT_FOUND) {
				cv::Vec3b ac = alice(y, x);
				cv::Vec3b bc = bob(y, x);
				visual(y, x) = cv::Vec3b(0, bgrToGrey(bc), bgrToGrey(ac));
				residualArea ++;
			} else {
				cv::Vec3b mc = moved(y, x);
				cv::Vec3b bc = bob(y, x);
				visual(y, x) = cv::Vec3b(0, bgrToGrey(bc), bgrToGrey(mc));
				residualArea ++;
			}
		}
	}

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
