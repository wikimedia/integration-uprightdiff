CFLAGS +=-g -std=c++11 -Wall -O2
PREFIX=/usr

all: uprightdiff test

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

test:
	g++ $(CFLAGS) tests/RollingBlockCounterTest.cpp -lopencv_core -o test
	./test

clean:
	rm -f uprightdiff test
