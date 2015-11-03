#include <cstring>

#include "BlockMotionSearch.h"
#include "OutwardAlternatingSearch.h"

BlockMotionSearch::Mat1i BlockMotionSearch::search() {
	int yBlockCount = m_source.rows / m_blockSize;
	int xBlockCount = m_source.cols / m_blockSize;
	m_blockMotion = Mat1i(yBlockCount, xBlockCount, CV_32SC1);

	for (m_yIndex = 0; m_yIndex < yBlockCount; m_yIndex++) {
		m_y = m_yIndex * m_blockSize;
		for (m_xIndex = 0; m_xIndex < xBlockCount; m_xIndex++) {
			m_x = m_xIndex * m_blockSize;
			cv::Rect sourceRect(m_x, m_y, m_blockSize, m_blockSize);
			Mat3b sourceBlock = m_source(sourceRect);

			// Priority 1: exactly constant baseline
			if (m_xIndex > 0 && m_blockMotion(m_yIndex, m_xIndex - 1) != NOT_FOUND) {
				if (tryMotion(sourceBlock, m_blockMotion(m_yIndex, m_xIndex - 1))) {
					continue;
				}
			}

			int searchStart;
			if (m_yIndex > 0 && m_blockMotion(m_yIndex - 1, m_xIndex) != NOT_FOUND) {
				// Priority 2: near-constant vertical flow
				searchStart = m_y + m_blockMotion(m_yIndex - 1, m_xIndex);
			} else if (m_xIndex > 0 && m_blockMotion(m_yIndex, m_xIndex - 1) != NOT_FOUND) {
				// Priority 3: near-constant baseline
				searchStart = m_y + m_blockMotion(m_yIndex, m_xIndex - 1);
			} else {
				// Priority 4: source offset
				searchStart = m_y;
			}
			// Check bounds of searchStart
			if (searchStart >= m_dest.rows - m_blockSize) {
				searchStart = m_dest.rows - m_blockSize - 1;
			}
			if (searchStart < 0) {
				searchStart = 0;
			}

			// Make sure the search window includes the no-change case
			int tempWindowSize = std::max(std::abs(searchStart - m_y), m_windowSize);

			OutwardAlternatingSearch search(searchStart, m_dest.rows - m_blockSize, tempWindowSize);
			m_blockMotion(m_yIndex, m_xIndex) = NOT_FOUND;
			for (; search; ++search) {
				if (tryMotion(sourceBlock, search.pos() - m_y)) {
					break;
				}
			}
		}
	}
	return m_blockMotion;
}

bool BlockMotionSearch::tryMotion(const Mat3b & sourceBlock, int dy) {
	cv::Rect destRect(m_x, m_y + dy, m_blockSize, m_blockSize);
	Mat3b destBlock = m_dest(destRect);
	if (blockEqual(sourceBlock, destBlock)) {
		m_blockMotion(m_yIndex, m_xIndex) = dy;
		return true;
	} else {
		return false;
	}
}

bool BlockMotionSearch::blockEqual(const Mat3b & m1, const Mat3b & m2) {
	if (m1.size() != m2.size()) {
		return false;
	}
	int rowSize = m1.cols * m1.elemSize();
	for (int y = 0; y < m1.rows; y++) {
		cv::Mat_<unsigned char> row1 = m1.rowRange(y, y+1).reshape(1, 1);
		cv::Mat_<unsigned char> row2 = m2.rowRange(y, y+1).reshape(1, 1);
		if (std::memcmp(row1.ptr(), row2.ptr(), rowSize) != 0) {
			return false;
		}
	}
	return true;
}

