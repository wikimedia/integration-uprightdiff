This utility examines the differences between two images. It produces a visual
annotation and reports statistics.

It is optimised for images which come from browser screenshots. It analyses the
image for vertical motion, and annotates connected regions that have the same
vertical displacement. Then it highlights any remaining ("residual")
differences which are not explained by vertical motion on a pixel-by-pixel
basis.

## Usage

```
./uprightdiff [options] <input-1> <input-2> <output>
Accepted options are:
  --help                  Show help message and exit
  --block-size arg        Block size for initial search (default 16)
  --window-size arg       Initial range for vertical motion detection (default 
                          200)
  --brush-width arg       Brush width when heuristically expanding blocks. A 
                          higher value gives smoother motion regions. This 
                          should be an odd number. (default 9)
  --outer-hl-window arg   The size of the outer square used for detecting 
                          isolated small features to highlight. This size 
                          defines what we mean by "isolated". It should be an 
                          odd number. (default 21)
  --inner-hl-window arg   The size of the inner square used for detecting 
                          isolated small features to highlight. This size 
                          defines what we mean by "small". It should be an odd 
                          number. (default 5)
  --intermediate-dir arg  A directory where intermediate images should be 
                          placed. This is our equivalent of debug or trace 
                          output.
  -v [ --verbose ]        Write progress info to stderr.
  --format arg            The output format for statistics, may be text (the 
                          default), json or none.
  -t [ --log-timestamp ]  Annotate progress info with elapsed time.
```

If you see an error "libdc1394 error: Failed to initialize libdc1394", this can
be ignored. OpenCV links to libdc1394 unconditionally, and libdc1394
unconditionally writes out this error on startup if it does not find any
cameras, but this utility does not make use of any camera functions. This is
fixed in OpenCV 3.x (which is not released yet): libdc1394 is now optional.

## Algorithm description

Motion is detected by first doing an exhaustive search for motion of blocks,
16x16 pixels by default. The block size should be large enough so that a single
block by itself contains identifying features, but not so large that regions of
motion will be missed. This usually means that it should be similar to the font
size.

Unlike similar algorithms used by video compression or robotics, we require an
exact match for motion search to succeed.

Then, starting from the block search results, regions with known motion are
expanded into regions of unknown motion. This is done at the full resolution,
but with a broad "brush size", defaulting to 9px, which defines a minimum
region of action. Using a brush size which is similar to or larger than the
font size prevents motion regions from closely hugging the text.

Connected regions in the resulting optical flow map are identified by a flood
fill algorithm. If a region has an area of at least 50px, it is shown in the
annotation as a contour outline, with a labelled arrow showing the direction and
magnitude of motion. These labels use a rotating palette of bluish colours.
Regions smaller than 50px are simply filled with the bluish colour.

A model of the first image is created, with motion applied. Regions
where the second image matches the moved image are shown in grey, half faded
to white.

Residuals are shown by putting the intensity of the moved model in the red
channel, and the intensity of the second image in the green channel. If the
motion search algorithm failed for a given pixel, then the first image is used,
instead of the moved image. Thus, residual colours appear as follows:

| First | Second | Result
|-------|--------|-------
| Dark  | Dark   | Dark
| Dark  | Light  | Green
| Light | Dark   | Red
| Light | Light  | Yellow

Since it is hard to spot isolated residual pixels by eye, we finally run a
search for isolated residual features, and put a yellow circle around any that
are found. An isolated feature is defined as a location where all of the
residual pixels in a large outer square are also within a smaller inner square
at its centre.

The following statistics are provided:

* modifiedArea: This is a simple count of the number of pixels for which the
  source does not match the destination (after they have both been expanded to
  the same size).
* movedArea: The number of pixels for which nonzero motion was detected.
* residualArea: The number of pixels which differed between the resulting image
  and the second input image.

By specifying --format=json, these statistics are provided on stdout JSON format,
e.g.:

{"modifiedArea":5045596,"movedArea":6081096,"residualArea":78707}

## Compilation

Install the dependencies. On Debian/Ubuntu this means:

`sudo apt-get install build-essential g++ libopencv-highgui-dev libboost-program-options-dev`

Then compile:

`make`

And optionally install it:

`make install PREFIX=/usr/local`

All dependencies (C++11, Boost and OpenCV) are cross-platform, and the code is
intended to be portable, so it should be possible to compile it on other
operating systems if necessary. However, a custom makefile will be required.
