#!/bin/sh

relDir="."

if [ -n $1 ]; then
	relDir=$1;
fi

./test_win.sh $relDir/videoSmall.mp4 \
		&& ./test_win.sh $relDir/videoMedium.mp4 \
		&& ./test_win.sh $relDir/videoLarge.mp4 \
		&& ./test_win.sh $relDir/large.avi
