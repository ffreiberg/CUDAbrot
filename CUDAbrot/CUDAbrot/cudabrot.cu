#include "cudabrot.cuh"

using namespace std;

#define cudaAssertSuccess(ans) { _cudaAssertSuccess((ans), __FILE__, __LINE__); }
inline void _cudaAssertSuccess(cudaError_t err, char *file, int line)
{
	if (err != cudaSuccess)  {
		fprintf(stderr, "_cudaAssertSuccess: %s %s %d\n", cudaGetErrorString(err), file, line);
		exit(err);
	}
}

int main(void){
	
	cv::Mat image = cv::Mat(IMGHEIGHT, IMGWIDTH, CV_8UC(3));
	cv::Point mouseP;
	cv::namedWindow(WINDOW, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(WINDOW, CallBackFunc, &mouseP);
	int * imgBuf, zoom = 1;
	double xx = .0, yy = .0;

#ifdef GPU
	cudaMallocHost((void**)&imgBuf, IMGWIDTH * IMGHEIGHT * sizeof(int));
#elif CPU
	#ifdef _WIN32
		SYSTEMTIME start, end;
		int diff;
	#elif __linux __ || __unix || __APPLE__
	#endif
#endif

	while (true)
	{
#ifdef GPU
		callKernel(&imgBuf, xx, yy, zoom);
		
		for (int x = 0; x < IMGWIDTH; x++)	{
			for (int y = 0; y < IMGHEIGHT; y++) {
				image.at<cv::Vec3b>(cv::Point(x, y)) = colorizeBernstein(imgBuf[y * IMGWIDTH + x]);
			}
		}
#elif CPU
	#ifdef _WIN32
			GetSystemTime(&start);
	#elif __linux __ || __unix || __APPLE__
	#endif
	#if MANDELBROT
			mandelbrot(image, xx, yy, zoom);
	#elif JULIASET
			julia(image, xx, yy, zoom);
	#endif
	#ifdef _WIN32
			GetSystemTime(&end);
			diff = ((end.wSecond * 1000) + end.wMilliseconds) - ((start.wSecond * 1000) + start.wMilliseconds);
			cout << "Zoom factor " << zoom << ".0x" << endl;
			cout << "Calculation took " << diff << "ms on CPU" << endl;
	#elif __linux __ || __unix || __APPLE__
	#endif
#endif
		imshow(WINDOW, image);

		zoom *= 2;
		if (cv::waitKey(0) == 'q'){
			cout << "q button pressed, terminating execution..." << endl << "Press any key to confirm" << endl;
			cv::waitKey(0);
			break;
		}
		else if (zoom >= LONG_MAX / 2){
			cout << "maximum zoom reached, terminating execution..." << endl << "Press any key to confrim" << endl;
			cv::waitKey(0);
			break;
		}

#if MANDELBROT
		xx = transXM(mouseP.x, .0, 1);
		yy = transYM(mouseP.y, .0, 1);
#elif JULIASET
		xx = transXJ(mouseP.x, .0, 1);
		yy = transYJ(mouseP.y, .0, 1);
#endif

	}

	return 0;
}

/****************************************************************************************************
*																									*
*																									*
*																									*
****************************************************************************************************/

void callKernel(int** img, double xx, double yy, int zoom){
	int* d_image_buffer;
#ifdef _WIN32
	SYSTEMTIME start, end;
	int diff;
#elif __linux __ || __unix || __APPLE__
#endif
	cudaAssertSuccess(cudaMalloc(&d_image_buffer, IMGWIDTH * IMGHEIGHT*sizeof(int)));
	dim3 block_size(THREADSX, THREADSY);
	dim3 grid_size((IMGWIDTH + block_size.x - 1)/ block_size.x, (IMGHEIGHT + block_size.y - 1)/ block_size.y);

#ifdef _WIN32
	GetSystemTime(&start);
#elif __linux __ || __unix || __APPLE__
#endif

#if MANDELBROT
	mandelbrotKernel<<<grid_size, block_size>>>(d_image_buffer, xx, yy, zoom);
#elif JULIASET
	juliaKernel<<<grid_size, block_size>>>(d_image_buffer, xx, yy, zoom);
#endif
	cudaAssertSuccess(cudaDeviceSynchronize());

#ifdef _WIN32
	GetSystemTime(&end);
	diff = ((end.wSecond * 1000) + end.wMilliseconds) - ((start.wSecond * 1000) + start.wMilliseconds);
	cout << "Zoom factor " << zoom << ".0x" << endl;
	cout << "Calculation took " << diff << "ms on GPU" << endl;
#elif __linux __ || __unix || __APPLE__
#endif
	cudaAssertSuccess(cudaPeekAtLastError());
	cudaAssertSuccess(cudaDeviceSynchronize());
	cudaAssertSuccess(cudaMemcpy(*img, d_image_buffer, IMGHEIGHT * IMGWIDTH*(sizeof(int)), cudaMemcpyDeviceToHost));
	cudaAssertSuccess(cudaFree(d_image_buffer));
}

#pragma region mandelbrot
/****************************************************************************************************
*																									*
*																									*
*																									*
****************************************************************************************************/

__host__ __device__ double transXM(double pX, double xx, int zoom) {
	return ((((double)(pX) / IMGWIDTH) * 3.5 - 2.5) / zoom) + xx;
}

__host__ __device__ double transYM(double pY, double yy, int zoom) {
	return ((((double)(pY) / IMGHEIGHT) * 3.5 - 1.75) / zoom) + yy;
}

__global__ void mandelbrotKernel(int* imageBuffer, double xx, double yy, int zoom) {
	int pY = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
	int pX = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
	
	/* if pX oder pY is out of bounds return */
	if (pX >= IMGWIDTH || pY >= IMGHEIGHT) {
		return;
	}

	/* set idx to position x,y */
	int idx = pY * IMGWIDTH + pX;

	/* transform coordinates */
	double cRe = transXM(pX, xx, zoom);
	double cIm = transYM(pY, yy, zoom);

	double zRe = 0.0;
	double zIm = 0.0;
	int iter = 0;
	double zReNew;
	
	/*  */
	while ((zRe * zRe + zIm * zIm <= ESCAPE) && (iter < MAX_ITER))
	{
		zReNew = zRe * zRe - zIm * zIm + cRe;
		zIm = 2.0 * zRe * zIm + cIm;
		zRe = zReNew;
		iter++;
	}

	imageBuffer[idx] = iter;
}

void mandelbrot(cv::Mat &imgBuffer, double xx, double yy, int zoom){

	for (int pX = 0; pX < IMGWIDTH; ++pX)
	{
		for (int pY = 0; pY < IMGHEIGHT; ++pY)
		{
			/* transform coordinates */
			double cRe = transXM(pX, xx, zoom);
			double cIm = transYM(pY, yy, zoom);

			double zRe = 0.0;
			double zIm = 0.0;
			int iter = 0;
			double zReNew;

			/*  */
			while ((zRe * zRe + zIm * zIm <= ESCAPE) && (iter < MAX_ITER))
			{
				zReNew = zRe * zRe - zIm * zIm + cRe;
				zIm = 2.0 * zRe * zIm + cIm;
				zRe = zReNew;
				iter++;
			}

			cv::Vec3b color = colorizeBernstein(iter);
			imgBuffer.at<cv::Vec3b>(cv::Point(pX, pY)) = color;
		}
	}
}
#pragma endregion

#pragma region juliaset
/****************************************************************************************************
*																									*
*																									*
*																									*
****************************************************************************************************/

__host__ __device__ double transXJ(double pX, double xx, int zoom){
	return  1.7 * (double)(IMGWIDTH / 2 - pX) / (IMGWIDTH / 2 * zoom) + xx;
}

__host__ __device__ double transYJ(double pY, double yy, int zoom){
	return 1.7 * (double)(IMGHEIGHT / 2 - pY) / (IMGHEIGHT / 2 * zoom) + yy;
}

__global__ void juliaKernel(int* imageBuffer, double xx, double yy, int zoom) {
	int pY = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
	int pX = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT

	/* if pX oder pY is out of bounds return */
	if (pX >= IMGWIDTH || pY >= IMGHEIGHT) {
		return;
	}

	/* set idx to position x,y */
	int idx = pY * IMGWIDTH + pX;

	/* transform coordinates */
	double zRe = transXJ(pX, xx, zoom);
	double zIm = transYJ(pY, yy, zoom);

	double cRe = -0.8;
	double cIm = 0.156;
	int iter = 0;
	double zReNew;

	while ((zRe * zRe + zIm * zIm <= ESCAPE) && (iter < MAX_ITER))
	{
		zReNew = zRe * zRe - zIm * zIm + cRe;
		zIm = 2.0 * zRe * zIm + cIm;
		zRe = zReNew;
		iter++;
	}

	imageBuffer[idx] = iter;
}

void julia(cv::Mat& imgBuffer, double xx, double yy, int zoom) {
	for (int pX = 0; pX < IMGWIDTH; ++pX)
	{
		for (int pY = 0; pY < IMGHEIGHT; ++pY)
		{
			/* transform coordinates */
			double zRe = transXJ(pX, xx, zoom);
			double zIm = transYJ(pY, yy, zoom);

			double cRe = -0.8;
			double cIm = 0.156;
			int iter = 0;
			double zReNew;

			while ((zRe * zRe + zIm * zIm <= ESCAPE) && (iter < MAX_ITER))
			{
				zReNew = zRe * zRe - zIm * zIm + cRe;
				zIm = 2.0 * zRe * zIm + cIm;
				zRe = zReNew;
				iter++;
			}
			cv::Vec3b color = colorizeBernstein(iter);
			imgBuffer.at<cv::Vec3b>(cv::Point(pX, pY)) = color;
		}
	}
}
#pragma endregion

#pragma region colorize
/****************************************************************************************************
*																									*
*																									*
*																									*
****************************************************************************************************/

cv::Vec3b colorizeBernstein(int val){

	double t = (double)val / (double)MAX_ITER;

	// Use smooth polynomials for r, g, b
	int r = (int)(9 * (1 - t)*t*t*t * 255);
	int g = (int)(15 * (1 - t)*(1 - t)*t*t * 255);
	int b = (int)(8.5*(1 - t)*(1 - t)*(1 - t)*t * 255);
	return cv::Vec3b(g, b, r);
}

cv::Vec3b colorizeEsoteric(int val){
	int N = 256; // colors per element
	int N3 = N * N * N;
	// map n on the 0..1 interval (real numbers)
	double t = (double)val / (double)MAX_ITER;
	// expand n on the 0 .. 256^3 interval (integers)
	val = (int)(t * (double)N3);

	int b = val / (N * N);
	int nn = val - b * N * N;
	int r = nn / N;
	int g = nn - r * N;
	return cv::Vec3b(g, b, r);
}
#pragma endregion

#pragma region callback
/****************************************************************************************************
*																									*
*																									*
*																									*
****************************************************************************************************/
void CallBackFunc(int event, int x, int y, int flags, void* ptr){
	if (event == cv::EVENT_LBUTTONDOWN){
		cv::Point *p = (cv::Point*)ptr;
		p->x = x;
		p->y = y;
		cout << "left clicked at (" << x << ", " << y << ")" << endl;
	}
}
#pragma endregion