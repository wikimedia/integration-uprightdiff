CFLAGS=-g

imagediff:
	g++ $(CFLAGS) imagediff.cpp \
	   -lopencv_highgui \
	   -lopencv_core \
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
