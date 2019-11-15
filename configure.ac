#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([uprightdiff], [1.4.0], [https://phabricator.wikimedia.org/tag/uprightdiff/])

AM_INIT_AUTOMAKE
EXTRA_CFLAGS=[-g -std=c++11 -Wall -O2]

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

AC_LANG_PUSH([C++])

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lboost_program_options':
AC_CHECK_LIB([boost_program_options], [main])
# FIXME: Replace `main' with a function in `-lopencv_core':
AC_CHECK_LIB([opencv_core], [main])
# FIXME: Replace `main' with a function in `-lopencv_highgui':
AC_CHECK_LIB([opencv_highgui], [main])
# FIXME: Replace `main' with a function in `-lopencv_imgproc':
AC_CHECK_LIB([opencv_imgproc], [main])
# FIXME: Replace `main' with a function in `-lopencv_imgcodecs':
AC_CHECK_LIB([opencv_imgcodecs], [main])

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions.

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
