CFLAGS=-g -std=c++11 -Wall -O2
PREFIX=/usr/local

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

linehash:
	g++ $(CFLAGS) linehash.cpp \
		-lmhash \
		-lopencv_highgui \
		-lopencv_core \
		-o linehash

fback:
	g++ $(CFLAGS) fback.cpp \
		-lopencv_highgui \
		-lopencv_core \
		-lopencv_video \
		-lopencv_imgproc \
		-o fback

clean:
	rm -f uprightdiff
