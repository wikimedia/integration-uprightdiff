#include <opencv2/core/core.hpp>

class BlockMotionSearch {
public:
	typedef cv::Mat_<cv::Vec3b> Mat3b;
	typedef cv::Mat_<int> Mat1i;

	enum {NOT_FOUND = 0x7fffffff};

	static Mat1i Search(const Mat3b & alice, const Mat3b & bob,
			int blockSize, int windowSize)
	{
		BlockMotionSearch obj(alice, bob, blockSize, windowSize);
		return obj.search();
	}

private:

	BlockMotionSearch(const Mat3b & alice, const Mat3b & bob,
			int blockSize, int windowSize)
		: m_source(alice), m_dest(bob), m_blockSize(blockSize), m_windowSize(windowSize)
	{}

	Mat1i search();
	bool tryMotion(const Mat3b & sourceBlock, int dy);
	bool blockEqual(const Mat3b & m1, const Mat3b & m2);

	const Mat3b & m_source;
	const Mat3b & m_dest;
	Mat1i m_blockMotion;
	const int m_blockSize;
	const int m_windowSize;

	int m_xIndex, m_yIndex, m_x, m_y;
};



