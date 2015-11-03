
class OutwardAlternatingSearch {
	public:
		OutwardAlternatingSearch(int middle, int height, int window)
			: m_middle(middle), m_height(height), m_window(window),
			m_negative(true), m_distance(0), m_done(false)
		{}

		const OutwardAlternatingSearch & operator++() {
			if (m_negative) {
				m_distance++;
				if (m_middle + m_distance < m_height) {
					m_negative = false;
				} else if (m_middle - m_distance < 0) {
					m_done = true;
				}
			} else {
				if (m_middle - m_distance < 0) {
					m_distance++;
					if (m_middle + m_distance >= m_height) {
						m_done = true;
					}
				} else {
					m_negative = true;
				}
			}
			if (m_distance > m_window) {
				m_done = true;
			}
		}

		operator bool() {
			return !m_done;
		}

		int offset() const {
			return m_negative ? - m_distance : m_distance;
		}

		int pos() const {
			return m_middle + offset();
		}

	private:
		int m_middle;
		int m_height;
		int m_window;
		bool m_negative;
		int m_distance;
		bool m_done;
};

