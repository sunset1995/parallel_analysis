#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>

#define smallestD 20.0

using namespace std;
using namespace cv;

struct BGR {
	int b, g, r;
	BGR(int b,int g,int r)
	:b(b), g(g), r(r) {}
	bool operator < (const BGR &rth) const {
		return b+g+r < rth.b+rth.g+rth.r;
	}
};

int main(int argc, const char** argv){
	if (CV_MAJOR_VERSION < 3) {
		puts("Advise you update to OpenCV3");
	}
	if( argc<2 ) {
		puts("Please specify input image path");
		return 0;
	}
	// Read an image
	// The 3 channels are BGR (Blue, Green, and Red)
	Mat img = imread(argv[1], 1);
	if( !img.data ) {
		puts("read image failed");
		return 0;
	}
	imshow("origin", img);
	int rows = img.rows;
	int cols = img.cols;

	clock_t cnt = clock();
	Mat YCrCb;
	cvtColor(img, YCrCb, CV_BGR2YCrCb);

	// enhence robustness
	double blockSumR[3][4], blockSumB[3][4];
	double blockSumDr[3][4], blockSumDb[3][4];
	int blockNum=0;
	for(int rid=0; rid<3; ++rid)
		for(int cid=0; cid<4; ++cid) {
			// get block info
			int rfrom = rid * (rows/3);
			int rto = (rid==2)? rows : rfrom + (rows/3);
			int cfrom = cid * (cols/4);
			int cto = (cid==3)? cols : cfrom + (cols/4);
			int sz = (rto - rfrom) * (cto - cfrom);

			// count Cr, Cb average of block
			blockSumR[rid][cid] = 0;
			blockSumB[rid][cid] = 0;
			for(int i=rfrom; i<rto; ++i)
				for(int j=cfrom; j<cto; ++j) {
					blockSumR[rid][cid] += YCrCb.at<Vec3b>(i,j)[1];
					blockSumB[rid][cid] += YCrCb.at<Vec3b>(i,j)[2];
				}
			double avgMr = blockSumR[rid][cid] / sz;
			double avgMb = blockSumB[rid][cid] / sz;

			// count Cr, Cb deviation of block
			blockSumDr[rid][cid] = 0;
			blockSumDb[rid][cid] = 0;
			for(int i=rfrom; i<rto; ++i)
				for(int j=cfrom; j<cto; ++j) {
					blockSumDr[rid][cid] += 
						fabs(YCrCb.at<Vec3b>(i,j)[1] - avgMr);
					blockSumDb[rid][cid] +=
						fabs(YCrCb.at<Vec3b>(i,j)[2] - avgMb);
				}

			// discard block if the deviation is too small
			double partDr = blockSumDr[rid][cid] / sz;
			double partDb = blockSumDb[rid][cid] / sz;
			if( partDr < smallestD && partDb < smallestD ) {
				blockSumDr[rid][cid] = 
				blockSumDb[rid][cid] =
				blockSumR[rid][cid] = 
				blockSumB[rid][cid] =  0;
			}
			else blockNum += sz;
		}
	printf("blockNum %d\n",blockNum);
	puts("sumR");
	for(int i=0; i<3; ++i, puts(""))
		for(int j=0; j<4; ++j)
			printf("%.2lf ",(double)blockSumR[i][j]); puts("");
	puts("sumB");
	for(int i=0; i<3; ++i, puts(""))
		for(int j=0; j<4; ++j)
			printf("%.2lf ",(double)blockSumB[i][j]); puts("");
	puts("sumDr");
	for(int i=0; i<3; ++i, puts(""))
		for(int j=0; j<4; ++j)
			printf("%.2lf ",blockSumDr[i][j]); puts("");
	puts("sumDb");
	for(int i=0; i<3; ++i, puts(""))
		for(int j=0; j<4; ++j)
			printf("%.2lf ",blockSumDb[i][j]); puts("");

	// count global Mr, Mb, Dr, Db
	double Mr=0, Mb=0, Dr=0, Db=0;
	for(int i=0; i<3; ++i)
		for(int j=0; j<4; ++j) {
			Mr += blockSumR[i][j];
			Mb += blockSumB[i][j];
			Dr += blockSumDr[i][j];
			Db += blockSumDb[i][j];
		}
	Mr /= blockNum;
	Mb /= blockNum;
	Dr /= blockNum;
	Db /= blockNum;

	printf("Mr: %.2f\n",Mr);
	printf("Mb: %.2f\n",Mb);
	printf("Dr: %.2f\n",Dr);
	printf("Db: %.2f\n",Db);

	// stored 10% top bright pixel: reference white point
	vector< pair<int, BGR> > refWhiteP; 
	int signMb=0, signMr=0;
	if( fabs(Mb)>1e-8 ) {
		if( Mb>0 ) signMb = 1;
		else signMb = -1;
	}
	if( fabs(Mr)>1e-8 ) {
		if( Mr>0 ) signMr = 1;
		else signMr = -1;
	}
	printf("R upperbound: %.2f\n",1.5 * Dr);
	printf("B upperbound: %.2f\n",1.5 * Db);
	for(int i=0; i<rows; ++i)
		for(int j=0; j<cols; ++j) {
			if( fabs(
				YCrCb.at<Vec3b>(i,j)[1]
				- (1.5*Mr + Dr*signMr) ) > 1.5 * Dr )
				continue;
			if( fabs(
				YCrCb.at<Vec3b>(i,j)[2]
				- (Mb + Db*signMb) ) > 1.5 * Db )
				continue;
			// push available pixel
			refWhiteP.emplace_back(
				make_pair(
					-YCrCb.at<Vec3b>(i,j)[0],
					BGR(
						img.at<Vec3b>(i,j)[0],
						img.at<Vec3b>(i,j)[1],
						img.at<Vec3b>(i,j)[2]
					)
				)
			);
		}

	if( refWhiteP.empty() ) {
		puts("Can not find white point. Do nothing!!");
		return 0;
	}
	
	// use 10% top bright pixel to count avg of bgr
	int bSum=0, gSum=0, rSum=0;
	sort(refWhiteP.begin(), refWhiteP.end());
	int pickNum = refWhiteP.size()/10 + 1;
	int yMax = 0;
	printf("top10%% num %d\n",pickNum);
	for(int i=0; i<pickNum; ++i) {
		bSum += refWhiteP[i].second.b;
		gSum += refWhiteP[i].second.g;
		rSum += refWhiteP[i].second.r;
		yMax = max(yMax, -refWhiteP[i].first);
	}
	double bAvg = (double)bSum / pickNum;
	double gAvg = (double)gSum / pickNum;
	double rAvg = (double)rSum / pickNum;

	printf("avg(b, g, r): %.0f %.0f %.0f\n",bAvg, gAvg, rAvg);
	printf("yMax %d\n",yMax);
	double bGain = yMax / bAvg;
	double gGain = yMax / gAvg;
	double rGain = yMax / rAvg;
	printf("gain(b, g, r): %.2f %.2f %.2f\n",bGain, gGain, rGain);

	// white balance img
	for(int i=0; i<rows; ++i)
		for(int j=0; j<cols; ++j) {
			img.at<Vec3b>(i,j)[0] = min(255, 
				(int)(bGain * img.at<Vec3b>(i,j)[0]));
			img.at<Vec3b>(i,j)[1] = min(255,
				(int)(gGain * img.at<Vec3b>(i,j)[1]));
			img.at<Vec3b>(i,j)[2] = min(255,
				(int)(rGain * img.at<Vec3b>(i,j)[2]));
		}

	cnt += clock() - cnt;
	printf("%fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC / 1000.0));

	imshow("white balance", img);
	waitKey(0);
	return 0;
}

