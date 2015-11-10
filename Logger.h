#include <ostream>
#include <ctime>

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
		char buf[100];
		std::time_t rawTime;
		time(&rawTime);
		struct std::tm * timeInfo = localtime(&rawTime);
		std::strftime(buf, 100, "%F %T ", timeInfo);
		return std::string(buf);
	}
	
private:
	int m_level;
	bool m_showTimestamp;
	LogStream m_realStream;
	LogStream m_devNull;
};
