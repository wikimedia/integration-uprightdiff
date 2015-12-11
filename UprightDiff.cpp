#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

#include "UprightDiff.h"
#include "BlockMotionSearch.h"
#include "RollingBlockCounter.h"

typedef UprightDiff::uchar uchar;
typedef UprightDiff::Mat3b Mat3b;
typedef UprightDiff::Mat1i Mat1i;
typedef UprightDiff::Mat1b Mat1b;

void UprightDiff::Diff(const cv::Mat & alice, const cv::Mat & bob, const Options & options,
		Output & output) {
	UprightDiff uprightDiff(alice, bob, options, output);
	uprightDiff.execute();
	uprightDiff.m_alice.release();
	uprightDiff.m_bob.release();
	uprightDiff.m_motion.release();
}

UprightDiff::UprightDiff(
		const cv::Mat & alice,
		const cv::Mat & bob,
		const Options & options,
		Output & output)
	: m_options(options), m_output(output), 
	m_logger(options.logStream ? *options.logStream : std::cerr,
			options.logLevel, options.logTimestamp)
{
	m_size = cv::Size(
			std::max(alice.cols, bob.cols),
			std::max(alice.rows, bob.rows));
	info() << "Extending both images to size " << m_size.width << "x" << m_size.height << "\n";

	m_alice = ConvertInput("first", alice, m_size);
	m_bob = ConvertInput("second", bob, m_size);
}

void UprightDiff::execute() {
	m_output.totalArea = m_size.area();
	calculateMaskArea();

	// Calculate block motion by exhaustive search
	info() << "Searching for motion...\n";
	Mat1i blockMotion = BlockMotionSearch::Search(m_bob, m_alice, 
			m_options.blockSize, m_options.windowSize);

	// Scale up block motion matrix
	m_motion = ScaleUpMotion(blockMotion, m_options.blockSize, m_size);
	intermediateOutput("prepaint", m_motion * 10 + 128);

	info() << "Expanding motion blocks\n";

	// Expand block motion into sub-block NOT_FOUND regions
	for (int y = 0; y < m_size.height; y++) {
		// Paint right
		paintSubBlockLine(cv::Point(0, y), cv::Point(1, 0));
		// Paint left
		paintSubBlockLine(cv::Point(m_size.width - 1, y), cv::Point(-1, 0));
	}
	for (int x = 0; x < m_size.width; x++) {
		// Paint down
		paintSubBlockLine(cv::Point(x, 0), cv::Point(0, 1));
		// Paint up
		paintSubBlockLine(cv::Point(x, m_size.height - 1), cv::Point(0, -1));
	}
	intermediateOutput("postpaint", m_motion * 10 + 128);

	info() << "Calculating residuals\n";

	visualizeResidual();

	info() << "Annotating motion\n";

	// Draw motion annotations
	annotateMotion();

	info() << "Done\n";
}

Mat3b UprightDiff::ConvertInput(const char * label, const cv::Mat & input, const cv::Size & size) {
	if (input.type() != CV_8UC3) {
		throw std::runtime_error(std::string("The ") + label +
				" image is invalid or has the wrong pixel type\n");
	}
	Mat3b ret(size, cv::Vec3b(128, 128, 128));
	input.copyTo(ret(cv::Rect(cv::Point(), input.size())));
	return ret;
}

void UprightDiff::calculateMaskArea() {
	Mat1b mask(m_size, 0);
	for (int y = 0; y < m_size.height; y++) {
		for (int x = 0; x < m_size.width; x++) {
			if (m_alice(y, x) != m_bob(y, x)) {
				mask(y, x) = 255;
			}
		}
	}
	intermediateOutput("mask", mask);
	m_output.maskArea = cv::countNonZero(mask);
}

Mat1i UprightDiff::ScaleUpMotion(Mat1i & blockMotion, int blockSize, const cv::Size & destSize) {
	Mat1i motion(destSize);
	Mat1i notFound(1, 1, NOT_FOUND);
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

void UprightDiff::paintSubBlockLine(const cv::Point & start, const cv::Point & step) {
	int halfWidth = (m_options.brushWidth - 1) / 2;
	cv::Point brushStep(step.y, step.x);
	cv::Point halfWidthVector = halfWidth * brushStep;
	cv::Point pos = start;
	cv::Rect bounds(cv::Point(), m_motion.size());
	int prevConsensus = NOT_FOUND;
	while (bounds.contains(pos)) {
		cv::Rect roiRect(pos - halfWidthVector, pos + halfWidthVector + cv::Point(1, 1));
		if ((roiRect & bounds) != roiRect) {
			break;
		}
		cv::Mat1i roiBlock = m_motion(roiRect);

		// Paint the current step
		if (prevConsensus != NOT_FOUND && prevConsensus != INVALID) {
			int curConsensus = GetWeakConsensus(roiBlock);
			if (curConsensus == NOT_FOUND || curConsensus == prevConsensus) {
				for (int b = -halfWidth; b <= halfWidth; b++) {
					cv::Point srcPos = pos + b * brushStep;
					cv::Point destPos = srcPos + cv::Point(0, prevConsensus);
					if (bounds.contains(destPos) && m_bob(srcPos) == m_alice(destPos)) {
						m_motion.at<int>(srcPos) = prevConsensus;
					}
				}
			}
		}

		prevConsensus = GetStrongConsensus(roiBlock);
		GetStrongConsensus(roiBlock);
		pos += step;
	}
}

uchar UprightDiff::BgrToGrey(const cv::Vec3b & bgr) {
	return cv::saturate_cast<uchar>(
			76 * bgr[2] / 255     // Blue
			+ 150 * bgr[1] / 255  // Green
			+ 29 * bgr[0] / 255); // Red
}

cv::Vec3b UprightDiff::BgrToFadedGreyBgr(const cv::Vec3b & bgr) {
	uchar value = 127 + BgrToGrey(bgr) / 2;
	return cv::Vec3b(value, value, value);
}

/**
 * Get the value of all elements in the block, or INVALID if they are not all
 * the same.
 */
int UprightDiff::GetStrongConsensus(const cv::Mat1i & block) {
	int consensus = block(0, 0);
	for (int y = 0; y < block.rows; y++) {
		for (int x = 0; x < block.cols; x++) {
			if (block(y, x) != consensus) {
				return INVALID;
			}
		}
	}
	return consensus;
}

/**
 * Get the value of all elements of the block, or INVALID if they are not all
 * the same, except for NOT_FOUND elements which are ignored. If all elements
 * are NOT_FOUND, NOT_FOUND is returned.
 */
int UprightDiff::GetWeakConsensus(const cv::Mat1i & block) {
	int consensus = NOT_FOUND;
	for (int y = 0; y < block.rows; y++) {
		for (int x = 0; x < block.cols; x++) {
			int v = block(y, x);
			if (v != consensus && v != NOT_FOUND) {
				return INVALID;
			}
			if (v != NOT_FOUND) {
				consensus = v;
			}
		}
	}
	return consensus;
}

Mat3b UprightDiff::visualizeResidual() {
	// Prepare moved image
	Mat3b moved(m_size, cv::Vec3b(255, 0, 255));
	m_output.movedArea = 0;
	for (int y = 0; y < m_size.height; y++) {
		for (int x = 0; x < m_size.width; x++) {
			int dy = m_motion(y, x);
			if (dy != NOT_FOUND) {
				if (dy != 0) {
					m_output.movedArea++;
				}
				if (y + dy >= moved.rows || y + dy < 0) {
					throw std::runtime_error(
						"Error: out of bounds: (" +
						std::to_string(x) +
						", " +
						std::to_string(y) +
						" + " +
						std::to_string(dy) +
						")\n");
				}
				moved(y, x) = m_alice(y + dy, x);
			}
		}
	}
	intermediateOutput("moved", moved);

	// Compute residual visualisation
	m_output.visual = Mat3b(m_size, cv::Vec3b(128, 128, 128));
	Mat3b & visual = m_output.visual;
	m_output.residualArea = 0;
	Mat1b residualMask(m_size, uchar(0));
	for (int y = 0; y < m_size.height; y++) {
		for (int x = 0; x < m_size.width; x++) {
			if (moved(y, x) == m_bob(y, x)) {
				visual(y, x) = BgrToFadedGreyBgr(moved(y, x));
			} else if (m_motion(y, x) == NOT_FOUND) {
				cv::Vec3b ac = m_alice(y, x);
				cv::Vec3b bc = m_bob(y, x);
				if (ac == bc) {
					visual(y, x) = BgrToFadedGreyBgr(ac);
				} else {
					visual(y, x) = cv::Vec3b(0, BgrToGrey(bc), BgrToGrey(ac));
					m_output.residualArea ++;
					residualMask(y, x) = 1;

				}
			} else {
				cv::Vec3b mc = moved(y, x);
				cv::Vec3b bc = m_bob(y, x);
				visual(y, x) = cv::Vec3b(0, BgrToGrey(bc), BgrToGrey(mc));
				m_output.residualArea ++;
				residualMask(y, x) = 1;
			}
		}
	}
	intermediateOutput("residual-mask", residualMask);
	intermediateOutput("plain-residual", visual);

	// Highlight isolated residual pixels
	// This is done by maintaining a count of the number of residual pixels in
	// two concentric blocks. As the block moves, we subtract the row that left
	// the block, and add the row that entered the block. This is done in
	// column-major order so that the rows being added or subtracted are
	// contiguous in memory.
	int ihw = m_options.innerHighlightWindow;
	int ihw2 = (ihw - 1) / 2;
	int ohw = m_options.outerHighlightWindow;
	for (int cx = 0; cx < m_size.width; cx++) {
		RollingBlockCounter<Mat1b> innerCounter(residualMask, cx, ihw);
		RollingBlockCounter<Mat1b> outerCounter(residualMask, cx, ohw);
		
		for (int cy = 0; cy < m_size.height; cy++) {
			int innerCount = innerCounter(cy);
			int outerCount = outerCounter(cy);
			if (innerCount != 0 && innerCount == outerCount) {
				cv::circle(visual, cv::Point(cx, cy),
						std::min(10, ihw * 2),
						cv::Scalar(0, 0xff, 0xff), 2);
				cv::Rect innerRect(
					cv::Point(
						std::max(cx - ihw2, 0),
						std::max(cy - ihw2, 0)
					),
					cv::Point(
						std::min(cx + ihw2 + 1, m_size.width),
						std::min(cy + ihw2 + 1, m_size.height)
					)
				);
				residualMask(innerRect) = 0;
				innerCounter.purge();
				outerCounter.purge();
			}
		}
	}
	intermediateOutput("circled-residual", visual);
	return visual;
}

void UprightDiff::annotateMotion() {
	Mat3b contourVis(m_output.visual.size(), cv::Vec3b());

	std::vector<cv::Scalar> palette;
	palette.push_back(cv::Scalar(0xff, 0x00, 0x00));
	palette.push_back(cv::Scalar(0xff, 0x80, 0x00));
	palette.push_back(cv::Scalar(0xff, 0x00, 0x80));
	int paletteIndex = 0;

	// Find motion regions by flood filling
	Mat1i motion(cv::Size(m_motion.cols + 2, m_motion.rows + 2), NOT_FOUND);
	m_motion.copyTo(motion(cv::Rect(cv::Point(1, 1), m_motion.size())));
	int regionIndex = 0;
	Mat1b mask(cv::Size(motion.cols + 2, motion.rows + 2), uchar(0));
	const int minArea = 50;
	for (int y = 1; y < motion.rows - 1; y++) {
		for (int x = 1; x < motion.cols - 1; x++) {
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
					| (2 << 8) // mask value
					| cv::FLOODFILL_MASK_ONLY
					);
			Mat1b currentMask = (mask == 2);
			if (area < minArea) {
				// Too small for contour, fill instead
				contourVis.setTo(palette[paletteIndex],
					currentMask(cv::Rect(cv::Point(2, 2), mask.size() - cv::Size(4, 4))));
				paletteIndex = (paletteIndex + 1) % palette.size();
			} else {
				// Draw arrow
				cv::Point centrePoint = FindMaskCentre(mask, area);
				cv::Scalar colour = palette[regionIndex % palette.size()];
				ArrowedLine(contourVis, centrePoint + cv::Point(0, currentMotion),
						centrePoint, colour);

				// Draw arrow label
				std::string text = std::to_string(std::abs(currentMotion));
				cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN,
						1, 1, nullptr);
				cv::putText(contourVis, text,
						centrePoint + cv::Point(2, currentMotion / 2 + textSize.height / 2),
						cv::FONT_HERSHEY_PLAIN,	1, colour);

				// Find and draw contours
				std::vector<std::vector<cv::Point>> contours;
				findContours(currentMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
				drawContours(contourVis, contours, -1, colour,
						1, 8, cv::noArray(), INT_MAX, cv::Point(-2, -2));
				regionIndex++;
			}
			// Mark done areas
			mask.setTo(1, currentMask);
		}
	}

	// Blend with destination
	Mat3b & visual = m_output.visual;
	for (int y = 0; y < visual.rows; y++) {
		for (int x = 0; x < visual.cols; x++) {
			if (contourVis(y, x) != cv::Vec3b()) {
				visual(y, x) = visual(y, x) / 2 + contourVis(y, x) / 2;
			}
		}
	}
}

cv::Point UprightDiff::FindMaskCentre(const Mat1b & mask, int totalArea) {
	int sumX = 0, sumY = 0;
	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask(y, x) == 2) {
				sumX += x;
				sumY += y;
			}
		}
	}
	return cv::Point(sumX / totalArea, sumY / totalArea);
}


/**
 * Draw an arrowed line, similar to cv::arrowedLine()
 */
void UprightDiff::ArrowedLine(Mat3b img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color,
           int thickness, int line_type, int shift, double tipLength)
{
	 // Factor to normalize the size of the tip depending on the length of the arrow
    const double tipSize = std::max(3.0, cv::norm(pt1-pt2)*tipLength);

	cv::line(img, pt1, pt2, color, thickness, line_type, shift);

    const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );

	cv::Point p(cvRound(pt2.x + tipSize * std::cos(angle + CV_PI / 4)),
        cvRound(pt2.y + tipSize * std::sin(angle + CV_PI / 4)));
	cv::line(img, p, pt2, color, thickness, line_type, shift);

    p.x = cvRound(pt2.x + tipSize * std::cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * std::sin(angle - CV_PI / 4));
	cv::line(img, p, pt2, color, thickness, line_type, shift);
}

void UprightDiff::intermediateOutput(const char* label, const cv::MatExpr & expr) {
	if (!m_options.intermediateDir.empty()) {
		intermediateOutput(label, cv::Mat(expr));
	}
}

void UprightDiff::intermediateOutput(const char* label, const cv::Mat & m) {
	if (!m_options.intermediateDir.empty()) {
		cv::imwrite(m_options.intermediateDir + "/" + label, m);
	}
}
