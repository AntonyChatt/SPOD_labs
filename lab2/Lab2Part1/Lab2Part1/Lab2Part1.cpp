#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include "omp.h"

#define var uchar

using namespace std;
using namespace cv;

const int ThreadsNumber = 8;
const int repeats = 10;

typedef Vec3b Pixel;

//convolution
var** conv2(var** img, int w[3][3], int rows, int cols, int bm = 3, int bn = 3) {
	int n1 = rows + bm - 1;
	int n2 = cols + bn - 1;
	var** result = (var**)malloc(n1 * sizeof(var*));
	for (int i = 0; i < n1; i++) {
		result[i] = (var*)malloc(n2 * sizeof(var*));
		for (int j = 0; j < n2; j++) {
			int sum = 0;
			for (int m = 0; m < bm; m++) {
				for (int n = 0; n < bn; n++) {
					int rm = i - m;
					int rn = j - n;
					if (rm >= 0 && rm < rows && rn >= 0 && rn < cols)
						sum += img[rm][rn] * w[m][n];
				}
			}
			sum = abs(sum);
			result[i][j] = sum;
			if (sum > 255) result[i][j] = 255;
		}
	}
	return result;
}

//convolution threads
var** conv2T(var** img, int w[3][3], int rows, int cols, int bm = 3, int bn = 3) {
	int i, j;

	int n1 = rows + bm - 1;
	int n2 = cols + bn - 1;
	var** result = (var**)malloc(n1 * sizeof(var*));

#pragma omp parallel for num_threads(ThreadsNumber) private (i, j)
	for (i = 0; i < n1; i++) {
		result[i] = (var*)malloc(n2 * sizeof(var*));
		for (j = 0; j < n2; j++) {
			int sum = 0;
			for (int m = 0; m < bm; m++) {
				for (int n = 0; n < bn; n++) {
					int rm = i - m;
					int rn = j - n;
					if (rm >= 0 && rm < rows && rn >= 0 && rn < cols)
						sum += img[rm][rn] * w[m][n];
				}
			}
			sum = abs(sum);
			result[i][j] = sum;
			if (sum > 255) result[i][j] = 255;
		}
	}
	return result;
}

//convert matrix to array
var **mat2array(Mat & src) {
	int rows = src.rows;
	int cols = src.cols;
	var** result = (var**)malloc(rows * sizeof(var*));

	for (int i = 0; i < rows; i++) {
		uchar* data = src.ptr<uchar>(i);
		result[i] = (var*)malloc(cols * sizeof(var*));
		for (int j = 0; j < cols; j++) {
			int temp = data[j];
			result[i][j] = temp;
		}
	}
	return result;
}

//convert matrix to array threads
var** mat2arrayT(Mat& src) {
	int i, j;

	int rows = src.rows;
	int cols = src.cols;
	var** result = (var**)malloc(rows * sizeof(var*));

#pragma omp parallel for num_threads(ThreadsNumber) private (i, j)
	for (i = 0; i < rows; i++) {
		uchar* data = src.ptr<uchar>(i);
		result[i] = (var*)malloc(cols * sizeof(var*));
		for (j = 0; j < cols; j++) {
			int temp = data[j];
			result[i][j] = temp;
		}
	}
	return result;
}

//convert arry to matrix
Mat array2mat(var** result, int rows, int cols) {
	Mat src = Mat(rows - 2, cols - 2, 0);
	for (int i = 2; i < rows; i++) {
		uchar* data = src.ptr<uchar>(i - 2);
		for (int j = 2; j < cols; j++) {
			data[j - 2] = result[i][j];
		}
	}
	return src;
}

//convert arry to matrix threads
Mat array2matT(var * *result, int rows, int cols) {
	int i, j;

	Mat src = Mat(rows - 2, cols - 2, 0);

#pragma omp parallel for num_threads(ThreadsNumber) private (i, j)
	for (i = 2; i < rows; i++) {
		uchar* data = src.ptr<uchar>(i - 2);
		for (j = 2; j < cols; j++) {
			data[j - 2] = result[i][j];
		}
	}
	return src;
}

Mat addxy(Mat& src1, Mat& src2) {
	int rows = src1.rows > src2.rows ? src1.rows : src2.rows;
	int cols = src1.cols > src2.cols ? src1.cols : src2.cols;
	Mat dst = Mat(rows, cols, 0);
	for (int i = 0; i < rows; i++) {
		uchar* data = dst.ptr<uchar>(i);
		uchar* d1 = src1.ptr<uchar>(i);
		uchar* d2 = src2.ptr<uchar>(i);
		for (int j = 0; j < cols; j++) {
			int sum = sqrt(d1[j] * d1[j] + d2[j] * d2[j]);
			if (sum > 255) data[j] = 255;
			else data[j] = sum;
		}
	}
	return dst;
}

Mat addxyT(Mat& src1, Mat& src2) {
	int i, j;

	int rows = src1.rows > src2.rows ? src1.rows : src2.rows;
	int cols = src1.cols > src2.cols ? src1.cols : src2.cols;
	Mat dst = Mat(rows, cols, 0);

#pragma omp parallel for num_threads(ThreadsNumber) private (i, j)
	for (i = 0; i < rows; i++) {
		uchar* data = dst.ptr<uchar>(i);
		uchar* d1 = src1.ptr<uchar>(i);
		uchar* d2 = src2.ptr<uchar>(i);
		for ( j = 0; j < cols; j++) {
			int sum = sqrt(d1[j] * d1[j] + d2[j] * d2[j]);
			if (sum > 255) data[j] = 255;
			else data[j] = sum;
		}
	}
	return dst;
}

void calculateSobel(Mat& img, Mat& result) {
	int sx[3][3] = {
	{ -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, 1 },
	};

	int sy[3][3] = {
	{ -1, -2, -1 },
	{ 0, 0, 0 },
	{ 1, 2, 1 },
	};

	Mat1i intens(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			intens[i][j] = (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2]) / 3;
		}

	var** array = mat2array(intens);
	var** gradx = conv2(array, sx, img.rows, img.cols);
	var** grady = conv2(array, sy, img.rows, img.cols);
	Mat src1 = array2mat(gradx, img.rows, img.cols);
	Mat src2 = array2mat(grady, img.rows, img.cols);

	result = addxy(src1, src2);

	//imshow("start", img);
	//imshow("gradient", result);

	for (int i = 0; i < img.rows; i++) {
		free(array[i]);
		free(gradx[i]);
		free(grady[i]);
	}
	free(array);
	free(gradx);
	free(grady);

	//waitKey(0);
}

void calculateSobelThread(Mat& img, Mat& result) {
	int sx[3][3] = {
	{ -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, 1 },
	};

	int sy[3][3] = {
	{ 1, 2, 1 },
	{ 0, 0, 0 },
	{ 1, 2, 1 },
	};

	var B, G, R = 0;

	Mat1i intens(img.rows, img.cols);

#pragma omp parallel for num_threads(ThreadsNumber) private (B,G,R)
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			B = img.at<Vec3b>(i, j)[0];
			G = img.at<Vec3b>(i, j)[1];
			R = img.at<Vec3b>(i, j)[2];

			intens[i][j] = (B+G+R) / 3;
		}

	var** array = mat2arrayT(intens);
	var** gradx = conv2T(array, sx, img.rows, img.cols);
	var** grady = conv2T(array, sy, img.rows, img.cols);
	Mat src1 = array2matT(gradx, img.rows, img.cols);
	Mat src2 = array2matT(grady, img.rows, img.cols);

	result = addxyT(src1, src2);

	//imshow("start", img);
	//imshow("gradient", result);

	for (int i = 0; i < img.rows; i++) {
		free(array[i]);
		free(gradx[i]);
		free(grady[i]);
	}
	free(array);
	free(gradx);
	free(grady);

	//waitKey(0);
}

int main()
{
	Mat image1024x768 = imread("Resorces/1024x768.jpg");
    Mat image1280x960 = imread("Resorces/1280x960.jpg");
    Mat image2048x1536 = imread("Resorces/2048x1536.jpg");
    Mat image7680x4320 = imread("Resorces/7680x4320.jpg");

	if (image1024x768.empty() || image1280x960.empty() || image2048x1536.empty() || image7680x4320.empty())
	{
		cout << "404 Not Found" << endl;
		waitKey(0);
		return -1;
	}

	Mat image1024x768Modifed;
	Mat image1280x960Modifed;
	Mat image2048x1536Modifed;
	Mat image7680x4320Modifed;

	chrono::steady_clock::time_point start;
	chrono::steady_clock::time_point end;

	cout << "---------------------------MULTI THREAD----------------------------" << endl;

	//SEVERAL
	vector<chrono::duration<float>> severalThreadDuration;

	// Image modification
	for (int n = 0; n < repeats; n++)
	{
		cout << ">";

		//start timer
		start = chrono::high_resolution_clock::now();

		calculateSobelThread(image1024x768, image1024x768Modifed);
		calculateSobelThread(image1280x960, image1280x960Modifed);
		calculateSobelThread(image2048x1536, image2048x1536Modifed);
		calculateSobelThread(image7680x4320, image7680x4320Modifed);

		//stop timer
		end = chrono::high_resolution_clock::now();
		severalThreadDuration.emplace_back(end - start);
	}

	cout << "===============================================================" << endl << endl;

	// Several threads time output
	float severalThreadDurationTime = 0;
	int i = 0;

	cout << "All several threads calculation times: ";

	for (auto& time : severalThreadDuration)
	{
		cout << endl << "Repeat " << i << ": " << time.count(); i++;
		severalThreadDurationTime += time.count();
	}

	cout << endl << "General time of several threads calculation    " << severalThreadDurationTime << endl;
	cout << "Average time of several threads calculation    " << severalThreadDurationTime / repeats << endl << endl;

	cout << "---------------------------ONE THREAD----------------------------" << endl;

	//SINGLE
	vector<chrono::duration<float>> oneThreadDuration;

	// Image modification
	for (int n = 0; n < repeats; n++)
	{
		cout << ">";

		//start timer
		start = chrono::high_resolution_clock::now(); //Точка для начала счета времени

		calculateSobel(image1024x768, image1024x768Modifed);
		calculateSobel(image1280x960, image1280x960Modifed);
		calculateSobel(image2048x1536, image2048x1536Modifed);
		calculateSobel(image7680x4320, image7680x4320Modifed);

		//stop timer
		end = chrono::high_resolution_clock::now();
		oneThreadDuration.emplace_back(end - start);
	}

	cout << "===============================================================" << endl << endl;

	// Single threads time output
	float oneThreadDurationTime = 0;
	i = 0;

	cout << "All one thread calculation times: ";

	for (auto& time : oneThreadDuration)
	{
		cout << endl << "Repeat " << i << ": " << time.count(); i++;
		oneThreadDurationTime += time.count();
	}
	cout << endl << "General time of one thread calculation: " << oneThreadDurationTime << endl;
	cout << "Average time of one thread calculation " << oneThreadDurationTime / repeats << endl << endl;

	imshow("Result image 1024x768", image1024x768Modifed);  waitKey(0);
	imshow("Result image 1280x960", image1280x960Modifed);  waitKey(0);
	imshow("Result image 2048x1536", image2048x1536Modifed);  waitKey(0);
	imshow("Result image 7680x4320", image7680x4320Modifed);  waitKey(0);

	imwrite("Resorces/M1024x768M.jpg", image1024x768Modifed);
	imwrite("Resorces/M1280x960M.jpg", image1280x960Modifed);
	imwrite("Resorces/M2048x1536M.jpg", image2048x1536Modifed);
	imwrite("Resorces/M7680x4320M.jpg", image7680x4320Modifed);

    return 0;
}


