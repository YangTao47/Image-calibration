#include<iostream>
#include<opencv2/opencv.hpp>
#include <string>
#include <io.h>
using namespace cv;
using namespace std;
Mat getRoi(Mat& copyImage);
const string path = "E:/seafile/Seafile/Ê¸Ó¡Ñù±¾Í¼¿â/6.5´ç_À¶A/È±Ê§Ê¸Ó¡";
int main()
{
	int i = 123;
	_finddata_t file_info;
	string current_path = path + "/*.bmp";
	intptr_t handle = _findfirst(current_path.c_str(), &file_info);
	if (handle == -1) return 1;
	do
	{
		Mat copyImage = imread(path + "/" + file_info.name);
		Mat dst = getRoi(copyImage);
		imwrite("D:/Image-Registration-master/test2/testBA/" + to_string(i) + ".png", dst);
		i++;
		
	}while (!_findnext(handle, &file_info));

	_findclose(handle);
	/*Mat imgA = imread("D:\\Image-Registration-master\\test2\\testP4\\1.png");
	Mat imgB = imread("D:\\Image-Registration-master\\test2\\testP4\\2.png");
	//Mat imgC = imread("D:\\Image-Registration-master\\test2\\testL1\\3.png");
	//Mat roiA = imgA(Rect(190, 200, 2500, 1400));
	//Mat roiB = imgB(Rect(4550, 1350, 300, 1100));
	resize(imgA, imgA, Size(400, 340));
	resize(imgB, imgB, Size(400, 340));
	//resize(imgC, imgC, Size(400, 340));
	cvtColor(imgA, imgA, CV_BGR2GRAY);
	//cvtColor(imgB, imgB, CV_BGR2GRAY);
	//cvtColor(imgC, imgC, CV_BGR2GRAY);
	imwrite("D:\\Image-Registration-master\\test2\\testP4\\1.jpg", imgA);
	imwrite("D:\\Image-Registration-master\\test2\\testP4\\2.jpg", imgB);
	//imwrite("D:\\Image-Registration-master\\test2\\testL1\\3.jpg", imgC);
	//getchar();*/
	//getchar();
	return 0;


}
Rect add(Rect& a, Rect& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.width += b.width;
	a.height += b.height;
	return a;
}

Mat getRoi(Mat& src)
{
	Mat copyImage;
	copyImage = src.clone();
	cvtColor(copyImage, copyImage, CV_BGR2HSV);
	inRange(copyImage, Scalar(70, 43, 46), Scalar(130, 255, 255), copyImage);
	Mat kernel = getStructuringElement(0, Size(3, 3));
	morphologyEx(copyImage, copyImage, MORPH_OPEN, kernel);
	vector<vector<Point>> contours;
	findContours(copyImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<Point> points;
	for (auto&p : contours)
	{
		for (auto&point : p)
		{
			points.push_back(point);
		}
	}
	
	Rect tmp = 	boundingRect(points);
	Rect a(50, 50, 100, 100);
	add(tmp, a);
	Mat dst = src(tmp).clone();

	
	return dst;
}