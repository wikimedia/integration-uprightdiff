CFLAGS=-g -Wall -O2
PREFIX=/usr/local

all: uprightdiff

install: all
	install -d $(PREFIX)/bin
	install -s uprightdiff $(PREFIX)/bin/uprightdiff

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
