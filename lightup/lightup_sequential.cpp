#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <algorithm>
#include <ctime>

using namespace std;
using namespace cv;


void Video() {
	// Setup video capture device
	// Link it to the first capture device
	VideoCapture captureVideo;
	captureVideo.open("videoSmall.mp4");

	Mat frameFromVideo;
	clock_t cnt = 0;
	while(true){
		captureVideo >> frameFromVideo;
		if( frameFromVideo.empty() ) break;
		//imshow("origin", frameFromVideo);

		clock_t last = clock();
		for (int i = 0; i < frameFromVideo.rows; i++)
			for (int j = 0; j < frameFromVideo.cols; j++)
				//for(int k=0; k<3; ++k)
				frameFromVideo.at<Vec3b>(i,j)[0] = min(
					2*frameFromVideo.at<Vec3b>(i,j)[0] + 5,
					255
				);
		cnt += clock() - last;
		//imshow("outputCamera", frameFromVideo);

		if( waitKey(30)>=0 ) break;
	}
	printf("%fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC/1000.0));
}

int main(int argc, const char** argv){
	if( CV_MAJOR_VERSION < 3 ) {
		puts("Advise you update to OpenCV3");
	}
	Video();
	return 0;
}

