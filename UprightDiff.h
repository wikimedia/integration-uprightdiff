#include <limits.h>
#include <iostream>
#include "Logger.h"

class UprightDiff {
public:
	typedef unsigned char uchar;
	typedef cv::Mat_<cv::Vec3b> Mat3b;
	typedef cv::Mat_<int> Mat1i;
	typedef cv::Mat_<uchar> Mat1b;

	struct Options {
		Options()
			: blockSize(16),
			windowSize(200),
			brushWidth(9),
			outerHighlightWindow(21),
			innerHighlightWindow(5),
			logStream(NULL),
			logLevel(Logger::FATAL),
			logTimestamp(false)
		{}

		int blockSize;
		int windowSize;
		int brushWidth;
		int outerHighlightWindow;
		int innerHighlightWindow;
		std::string intermediateDir;
		std::ostream * logStream;
		int logLevel;
		bool logTimestamp;
	};

	struct Output {
		Output()
			: totalArea(0),
			maskArea(0),
			movedArea(0),
			residualArea(0)
		{}

		int totalArea;
		int maskArea;
		int movedArea;
		int residualArea;
		Mat3b visual;
	};

	enum {
		NOT_FOUND = INT_MAX,
		INVALID = NOT_FOUND - 1
	};

	static void Diff(const cv::Mat & alice, const cv::Mat & bob, const Options & options,
			Output & output);

private:
	UprightDiff(const cv::Mat & alice, const cv::Mat & bob, const Options & options,
			Output & output);

	void execute();
	void calculateMaskArea();
	static Mat3b ConvertInput(const char * label, const cv::Mat & input, const cv::Size & size);
	static Mat1i ScaleUpMotion(Mat1i & blockMotion, int blockSize, const cv::Size & destSize);
	void paintSubBlockLine(const cv::Point & start, const cv::Point & step);
	static uchar BgrToGrey(const cv::Vec3b & bgr);
	static cv::Vec3b BgrToFadedGreyBgr(const cv::Vec3b & bgr);
	static int GetStrongConsensus(const cv::Mat1i & block);
	static int GetWeakConsensus(const cv::Mat1i & block);
	Mat3b visualizeResidual();
	void annotateMotion();
	static cv::Point FindMaskCentre(const Mat1b & mask, int totalArea);
	static void ArrowedLine(Mat3b img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color,
			   int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1);

	void intermediateOutput(const char* label, const cv::MatExpr & expr);
	void intermediateOutput(const char* label, const cv::Mat & m);
	
	Logger::LogStream & info() {
		return m_logger.log(Logger::INFO);
	}

	const Options & m_options;
	Output & m_output;
	Mat3b m_alice;
	Mat3b m_bob;
	Mat1i m_motion;
	cv::Size m_size;
	Logger m_logger;
};
