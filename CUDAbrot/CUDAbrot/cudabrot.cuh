#ifndef CUDABROT_CUH
#define CUDACROT_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#elif __linux__ || __unix || __APPLE__
#include <time.h>
#endif
#include <climits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define IMGHEIGHT	1000			//image height
#define IMGWIDTH	1000			//image width
#define MAXITER		1000			//maximum number of iterations
#define ESCAPE		4.0				//escape value
#define THREADSX	16				//number of threads/block in x-direction
#define THREADSY	16				//number of threads/block in y direction
#define WINDOW		"CUDAbrot"		//name of window to illustrate set


/* Functions to colorize*/
cv::Vec3b colorizeBernstein(int);
cv::Vec3b colorizeEsoteric(int);

/* Function to call kernel */
void callKernel(int**, double, double, int);

/* Mandeltbrot functions */
void mandelbrot(cv::Mat &, double, double, int);
__global__ void mandelbrotKernel(int*, double, double, int);
__host__ __device__ double transXM(double, double, int);
__host__ __device__ double transYM(double, double, int);

/* JuliaSet functions */
void julia(cv::Mat &, double, double, int);
__global__ void juliaKernel(int*, double, double, int);
__host__ __device__ double transXJ(double, double, int);
__host__ __device__ double transYJ(double, double, int);

/* Callback function for mouse event */
void CallBackFunc(int, int, int, int, void*);

#endif