#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <mhash.h>

int main(int argc, char** argv) {
	hashid algo = MHASH_MD5;
	int hashSize = mhash_get_block_size(algo);
	unsigned char hash[hashSize + 1];

	if (argc < 1) {
		return 1;
	} else if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <input.png>\n";
		return 1;
	}
	cv::Mat input = cv::imread(argv[1]);

	for (int y = 0; y < input.rows; y++) {
		MHASH hashState = mhash_init(algo);
		const unsigned char* rowPtr = input.ptr(y);
		mhash(hashState, 
				input.ptr(y),
				input.elemSize() * input.cols);
		mhash_deinit(hashState, hash);

		for (int i = 0; i < hashSize; i++) {
			std::cout << std::hex << std::setfill('0') << std::setw(2) <<
				int(hash[i]);
		}
		std::cout << std::endl;
	}
	return 0;
}
