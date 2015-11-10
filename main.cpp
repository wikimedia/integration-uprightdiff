#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <iostream>
#include <cmath>

#include "ImageDiff.h"

int main(int argc, char** argv) {
	if (argc < 4) {
		std::cerr << "Usage: imagediff <input1> <input2> <output>\n";
		return 1;
	}

	ImageDiff::Options options;
	options.logLevel = Logger::INFO;

	cv::Mat alice = cv::imread(argv[1]);
	cv::Mat bob = cv::imread(argv[2]);
	ImageDiff::Output output;

	try {
		ImageDiff::Diff(alice, bob, options, output);
	} catch (std::runtime_error & e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
	cv::imwrite(argv[3], output.visual);

	std::cout << "Modified area: " << output.maskArea << " pixels\n";
	std::cout << "Moved area: " << output.movedArea << " pixels\n";
	std::cout << "Residual area: " << output.residualArea << " pixels\n";
}


