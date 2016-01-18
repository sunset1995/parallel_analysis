#!/bin/sh

relDir="."

if [ -n $1 ]; then
	relDir=$1;
fi

./test_lightup.sh $relDir/videoSmall.mp4 \
		&& ./test_lightup.sh $relDir/videoMedium.mp4 \
		&& ./test_lightup.sh $relDir/videoLarge.mp4 \
		&& ./test_lightup.sh $relDir/large.avi
