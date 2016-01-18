
#include "cuda_runtime.h"
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <ctime>

using namespace std;

__global__ void blue(unsigned short *mat, const int sz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sz) {
		unsigned short mi = mat[i];
		for (int k = 0; k < 1024; k++)
			mi = mi + (mi >> 3);
		mat[i] = mi;
	}
}


int Video() {
	// Setup video capture device
	// Link it to the first capture device
	cudaError_t err;

	int i, j;
	clock_t cnt = 0, cnt_io = 0;

	unsigned short *hmat[3], *dmat = NULL;
	int rows = 1920, cols = 1080, sz = rows * cols;
	int size = rows * cols * sizeof(unsigned short);
	unsigned short val[3] = { 0, 127, 255 };

	for (int k = 0; k < 3; k++) {
		hmat[k] = (unsigned short *)malloc(size);
		for (i = 0; i < rows; i++)
			for (j = 0; j < cols; j++)
				hmat[k][i * cols + j] = val[k];
	}

	clock_t last = clock();
	for (int k = 0; k < 3; k++) {
		dmat = NULL;
		err = cudaMalloc(&dmat, size); if (err != cudaSuccess) { puts("Error!"); return 1; }
		err = cudaMemcpy(dmat, hmat[k], size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed while Memcpying ! %s\n", cudaGetErrorString(err));
			return 1;
		}

		int blocks = (rows * cols + 1023) / 1024;
		blue << <blocks, 1024>> >(dmat, sz);
		cnt += clock() - last;
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed launching kernel! %s\n", cudaGetErrorString(err));
			return 1;
		}

		err = cudaMemcpy(hmat[k], dmat, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed while Memcpying back! %s\n", cudaGetErrorString(err));
			return 1;
		}

		cudaFree(dmat);

	}

	cnt += clock() - last;


//	cout << endl << "Results from frame 0: " << endl;
//	for (int k = 0; k < 3; ++k)
//		for (int i = 0; i < 256; ++i)
//			for (int j = 0; j < 256; ++j)
//				cout << hmat[k][i * cols + j] << " ";

	printf("Total = %fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC / 1000.0));

	for (int k = 0; k < 3; k++) free(hmat[k]);

	return 0;
}

int main()
{
	Video();
	while (1);
	return 0;
}



/*// Small
#include "cuda_runtime.h"
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <ctime>

using namespace std;

__global__ void blue(unsigned short *mat, int sz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sz) {
		unsigned short mi = mat[i];
		for (int k = 0; k < 1024; k++)
			mi = mi + (mi >> 3);
		mat[i] = mi;
	}
}

int Video() {
	// Setup video capture device
	// Link it to the first capture device
	cudaError_t err;

	int i, j;
	clock_t cnt = 0, cnt_io = 0;

	unsigned short *hmat, *dmat = NULL;



	int rows = 256, cols = 256;
	int size = rows * cols * sizeof(unsigned short);
	hmat = (unsigned short *)malloc(size);
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
			hmat[i * cols + j] = i * cols + j;


	clock_t last = clock();

	dmat = NULL;
	err = cudaMalloc(&dmat, size); if (err != cudaSuccess) { puts("Error!"); return 1; }
	err = cudaMemcpy(dmat, hmat, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed while Memcpying ! %s\n", cudaGetErrorString(err));
		return 1;
	}

	int blocks = (rows * cols + 1023) / 1024, sz = rows * cols;
	blue << <blocks, 1024>> >(dmat, sz);
	cnt += clock() - last;
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed launching kernel! %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(hmat, dmat, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed while Memcpying back! %s\n", cudaGetErrorString(err));
		return 1;
	}

	cnt += clock() - last;


	cout << endl << "Results from frame 0: " << endl;
	// for (int i = 0; i < 256; ++i)
	// 	for (int j = 0; j < 256; ++j)
	//		cout << hmat[i * cols + j] << " ";

	printf("Total = %fms\n", 1.0*cnt / (1.0*CLOCKS_PER_SEC / 1000.0));

	cudaFree(dmat);
	free(hmat);

	return 0;
}

int main()
{
	Video();
	while (1);
	return 0;
}
*/