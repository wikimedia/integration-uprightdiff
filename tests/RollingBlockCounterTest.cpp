#include <iostream>
#include <opencv2/core/core.hpp>
#include "../RollingBlockCounter.h"

typedef cv::Mat_<int> Mat1i;
bool good = true;

void rollingCount(const char* message, const Mat1i & mat, int window,
		bool purge, int * expectedData)
{
	Mat1i expected(mat.rows, mat.cols, expectedData);
	for (int x = 0; x < mat.cols; x++) {
		RollingBlockCounter<Mat1i> rbc(mat, x, window);
		for (int y = 0; y < mat.rows; y++) {
			if (rbc(y) != expected(y, x)) {
				std::cout << "Error: " << message << ": " << "at (" <<
					y << ", " << x << ") got " << rbc(y) <<
					", expected " << expected(y, x) << "\n";
				good = false;
			}
			if (purge) {
				rbc.purge();
			}
		}
	}
}

int main(int argc, char** argv) {
	int input[] = {
		50, 62, 61, 89, 15,
		71, 73, 69, 86, 72,
		78, 30, 89, 60, 64,
		22, 22, 99, 97, 93,
		43, 77, 75, 42, 13
	};

	int sum3[] = {
		256, 386, 440, 392, 262,
		364, 583, 619, 605, 386,
		296, 553, 625, 729, 472,
		272, 535, 591, 632, 369,
		164, 338, 412, 419, 245
	};

	int sum5[] = {
		583,  818,  969,  770,  605,
		726, 1058, 1302, 1081,  894,
		921, 1295, 1552, 1288, 1024,
		748, 1033, 1275, 1061,  859,
		535,  734,  904,  761,  632
	};

	Mat1i mat(5, 5, input);

	rollingCount("1-n", mat, 1, false, input);
	rollingCount("1-p", mat, 1, true, input);
	rollingCount("3-n", mat, 3, false, sum3);
	rollingCount("3-p", mat, 3, true, sum3);
	rollingCount("5-n", mat, 5, false, sum5);
	rollingCount("5-p", mat, 5, false, sum5);

	return good ? 0 : 1;
}
