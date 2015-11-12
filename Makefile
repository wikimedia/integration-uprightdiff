CFLAGS=-g -std=c++11 -Wall

imagediff:
	g++ $(CFLAGS) main.cpp BlockMotionSearch.cpp ImageDiff.cpp \
		-lopencv_highgui \
		-lopencv_core \
		-lopencv_imgproc \
		-lboost_program_options \
		-o imagediff

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
