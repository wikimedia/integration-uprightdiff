all:
	g++ imagediff.cpp \
	   -lopencv_highgui \
	   -lopencv_imgproc \
	   -lopencv_core \
	   -o imagediff
