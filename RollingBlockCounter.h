#include <opencv2/core/core.hpp>

template <class Mat>
class RollingBlockCounter {
public:
	RollingBlockCounter(const Mat & mat, int centreX, int window);
	void purge() {
		m_valid = false;
	}
	int operator ()(int centreY);
private:
	inline static cv::Rect GetBounds(int width, int height, int centreX, int window);

	int m_halfWindow;
	Mat m_strip;
	int m_cy;
	bool m_valid;
	int m_count;
};

template <class Mat>
RollingBlockCounter<Mat>::RollingBlockCounter(const Mat & mat, int cx, int window)
	: m_halfWindow((window - 1) / 2),
	m_strip(mat, GetBounds(mat.cols, mat.rows, cx, window)),
	m_cy(0), m_valid(false), m_count(0)
{}

template <class Mat>
cv::Rect RollingBlockCounter<Mat>::GetBounds(int width, int height, int cx, int window) {
	int halfWindow = (window - 1) / 2;
	int left = std::max(cx - halfWindow, 0);
	int right = std::min(cx + halfWindow + 1, width);
	return cv::Rect(left, 0, right - left, height);
}

template <class Mat>
int RollingBlockCounter<Mat>::operator ()(int cy) {
	int delta = 0;
	if (m_valid && cy == m_cy) {
		// No update needed
	} else if (m_valid && cy == m_cy + 1) {
		// Incremental calculation
		// Subtract top row (if any)
		int topY = cy - m_halfWindow - 1;
		if (topY >= 0) {
			auto * row = m_strip[topY];
			for (int x = 0; x < m_strip.cols; x++) {
				delta -= row[x];
			}
		}
		// Add bottom row
		int bottomY = cy + m_halfWindow;
		if (bottomY < m_strip.rows) {
			auto * row = m_strip[bottomY];
			for (int x = 0; x < m_strip.cols; x++) {
				delta += row[x];
			}
		}
		m_count += delta;
	} else {
		// Calculate from scratch
		int topY = std::max(cy - m_halfWindow, 0);
		int bottomY = std::min(cy + m_halfWindow, m_strip.rows - 1);
		for (int y = topY; y <= bottomY; y++) {
			auto * row = m_strip[y];
			for (int x = 0; x < m_strip.cols; x++) {
				delta += row[x];
			}
		}
		m_count = delta;
	}
	m_valid = true;
	m_cy = cy;
	return m_count;
}

