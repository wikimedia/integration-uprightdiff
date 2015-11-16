#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "UprightDiff.h"

namespace po = boost::program_options;

struct MainOptions {
	enum {
		NONE,
		TEXT,
		JSON
	} format = TEXT;

	std::string aliceName;
	std::string bobName;
	std::string destName;
};
bool processCommandLine(int argc, char** argv,
		MainOptions & mainOptions, UprightDiff::Options & diffOptions);

int main(int argc, char** argv) {
	MainOptions mainOptions;
	UprightDiff::Options diffOptions;

	try {
		if (!processCommandLine(argc, argv, mainOptions, diffOptions)) {
			return 1;
		}
	} catch (po::error & e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}

	cv::Mat alice = cv::imread(mainOptions.aliceName);
	cv::Mat bob = cv::imread(mainOptions.bobName);
	UprightDiff::Output output;

	try {
		UprightDiff::Diff(alice, bob, diffOptions, output);
	} catch (std::runtime_error & e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
	cv::imwrite(mainOptions.destName, output.visual);

	if (mainOptions.format == MainOptions::TEXT) {
		std::cout << "Modified area: " << output.maskArea << " pixels\n";
		std::cout << "Moved area: " << output.movedArea << " pixels\n";
		std::cout << "Residual area: " << output.residualArea << " pixels\n";
	} else if (mainOptions.format == MainOptions::JSON) {
		std::cout <<
			"{\"modifiedArea\":" << output.maskArea << "," <<
			"\"movedArea\":" << output.movedArea << "," <<
			"\"residualArea\":" << output.residualArea << "}\n";
	}
}

bool processCommandLine(int argc, char** argv,
		MainOptions & mainOptions, UprightDiff::Options & diffOptions)
{
	po::options_description visible;
	std::string format;
	visible.add_options()
		("help",
		 	"Show help message and exit")
		("block-size", po::value<int>(&diffOptions.blockSize),
			"Block size for initial search")
		("window-size", po::value<int>(&diffOptions.windowSize),
			"Initial range for vertical motion detection")
		("brush-width", po::value<int>(&diffOptions.brushWidth),
		 	"Brush width when heuristically expanding blocks. "
			"A higher value gives smoother motion regions. "
			"This should be an odd number.")
		("outer-hl-window", po::value<int>(&diffOptions.outerHighlightWindow),
			"The size of the outer square used for detecting isolated small features to highlight. "
			"This size defines what we mean by \"isolated\". It should be an odd number.")
		("inner-hl-window", po::value<int>(&diffOptions.innerHighlightWindow),
		 	"The size of the inner square used for detecting isolated small features to highlight. "
			"This size defines what we mean by \"small\". It should be an odd number.")
		("intermediate-dir", po::value<std::string>(&diffOptions.intermediateDir),
		 	"A directory where intermediate images should be placed. "
			"This is our equivalent of debug or trace output.")
		("verbose,v",
		 	"Write progress info to stderr.")
		("format", po::value<std::string>(&format),
		 	"The output format for statistics, may be text (the default), json or none.")
		("log-timestamp", po::bool_switch(&diffOptions.logTimestamp),
		 	"Annotate progress info with timestamps.")
		;

	po::options_description invisible;
	invisible.add_options()
		("alice", po::value<std::string>(&mainOptions.aliceName))
		("bob", po::value<std::string>(&mainOptions.bobName))
		("dest", po::value<std::string>(&mainOptions.destName))
		;

	po::options_description allDesc;
	allDesc.add(visible).add(invisible);

	po::positional_options_description positionalDesc;
	positionalDesc
		.add("alice", 1)
		.add("bob", 1)
		.add("dest", 1)
		;

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv)
			.options(allDesc)
			.positional(positionalDesc)
			.run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << "Usage: " << (argc >= 1 ? argv[0] : "uprightdiff" )
			<< " [options] <input-1> <input-2> <output>\n"
			<< "Accepted options are:\n"
			<< visible;
		return false;
	}
	if (vm.count("verbose")) {
		diffOptions.logLevel = Logger::INFO;
	}
	if (!(vm.count("alice") && vm.count("bob") && vm.count("dest"))) {
		std::cerr << "Error: two input filenames and an output filename must be specified.\n";
		return false;
	}
	if (vm.count("format")) {
		if (format == "text") {
			mainOptions.format = MainOptions::TEXT;
		} else if (format == "json") {
			mainOptions.format = MainOptions::JSON;
		} else if (format == "none") {
			mainOptions.format = MainOptions::NONE;
		} else {
			std::cerr << "Error: --format must be text, json or none\n";
			return false;
		}
	}

	return true;
}
