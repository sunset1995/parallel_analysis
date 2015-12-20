#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <algorithm>

using namespace std;
using namespace cv;


void Video() {
	// Setup video capture device
	// Link it to the first capture device
	VideoCapture captureVideo;
	captureVideo.open("videoSmall.mp4");

	Mat frameFromVideo;
	while(true){
		captureVideo >> frameFromVideo;
		if( frameFromVideo.empty() ) break;

		for (int i = 0; i < frameFromVideo.rows; i++)
			for (int j = 0; j < frameFromVideo.cols; j++)
				//for(int k=0; k<3; ++k)
				frameFromVideo.at<Vec3b>(i,j)[0] = min(
					2*frameFromVideo.at<Vec3b>(i,j)[0] + 5,
					255
				);
		imshow("outputCamera", frameFromVideo);

		if( waitKey(30)>=0 ) break;
	}
}

int main(int argc, const char** argv){
	if( CV_MAJOR_VERSION < 3 ) {
		puts("Advise you update to OpenCV3");
	}
	Video();
	return 0;
}

