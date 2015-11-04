CFLAGS=-g -std=c++11

imagediff:
	g++ $(CFLAGS) imagediff.cpp BlockMotionSearch.cpp \
		-lopencv_highgui \
		-lopencv_core \
		-lopencv_imgproc \
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
