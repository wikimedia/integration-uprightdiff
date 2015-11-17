#include <ostream>
#include <ctime>
#include <sstream>
#include <iomanip>

class Logger {
public:
	Logger(std::ostream & backend, int level, bool showTimestamp = false)
		: m_level(level), m_showTimestamp(showTimestamp),
		m_realStream(backend, true), m_devNull(backend, false)
	{}

	enum {TRACE, DEBUG, INFO, WARNING, ERROR, FATAL};

	class LogStream {
		friend class Logger;

		LogStream(std::ostream & backend, bool enabled)
			: m_backend(backend), m_enabled(enabled)
		{}

		std::ostream & m_backend;
		bool m_enabled;

	public:
		template <class T>
		LogStream & operator<<(const T & x) {
			if (m_enabled) {
				m_backend << x;
			}
			return *this;
		}

	};

	LogStream & log(int level) {
		if (level >= m_level) {
			if (m_showTimestamp) {
				m_realStream << timestamp();
			}
			return m_realStream;
		} else {
			return m_devNull;
		}
	}

	std::string timestamp() {
		std::clock_t now = clock();
		std::clock_t totalMillis = now / (CLOCKS_PER_SEC / 1000);

		std::ostringstream buf;
		buf << std::setw(3) << (totalMillis / 1000) << "."
		    << std::setfill('0') << std::setw(3) << (totalMillis % 1000)
			<< " ";
		return buf.str();
	}

private:
	int m_level;
	bool m_showTimestamp;
	LogStream m_realStream;
	LogStream m_devNull;
};
