CFLAGS +=-g -std=c++11 -Wall -O2
PREFIX=/usr

all: uprightdiff

install: all
	install -d $(DESTDIR)$(PREFIX)/bin
	install -s uprightdiff $(DESTDIR)$(PREFIX)/bin/uprightdiff

uprightdiff:
	g++ $(CFLAGS) main.cpp BlockMotionSearch.cpp UprightDiff.cpp \
		-lopencv_highgui \
		-lopencv_core \
		-lopencv_imgproc \
		-lboost_program_options \
		-o uprightdiff

clean:
	rm -f uprightdiff
