## Process this file with automake to produce Makefile.in
AUTOMAKE_OPTIONS = foreign
bin_PROGRAMS = uprightdiff
uprightdiff_SOURCES = main.cpp BlockMotionSearch.cpp UprightDiff.cpp

test:
	g++ $(CFLAGS) tests/RollingBlockCounterTest.cpp -lopencv_core -o test
	./test
