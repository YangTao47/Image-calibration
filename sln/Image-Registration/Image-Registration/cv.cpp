#include"Sar_sift.h"
#include"match.h"

#include <opencv2/core/core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\features2d\features2d.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include<opencv2\xfeatures2d\nonfree.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<opencv2\xfeatures2d.hpp>
#include<sstream>
#include<vector>
#include<fstream>
#include<io.h>


Mat imageRegistration(Mat& image_1, Mat image_2, string change_model = string("affine"))
{

	if (!image_1.data || !image_2.data){
		cout << "图像数据加载失败！" << endl;
		return Mat();
	}

	double total_beg = (double)getTickCount();//总时间开始

	Ptr<xfeatures2d::BriefDescriptorExtractor> des = xfeatures2d::BriefDescriptorExtractor::create();
	//参考图像特征点检测与描述
	Mat descriptors_1;
	vector<KeyPoint> keypoints_1;
	vector<Mat> sar_harris_fun_1, amplit_1, orient_1;
	double detect1_beg = (double)getTickCount();

	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(10,true);
	//Ptr<ORB> orb = ORB::create(500, pow(2, 1.0 / 3.0), 2, 10, 0, 2, ORB::HARRIS_SCORE, 10, 10);
	//Ptr<MSER> mser = MSER::create(2, 5, 100, 0.5, 0.3);
	Ptr<xfeatures2d::SIFT> sift1 = xfeatures2d::SIFT::create(500, 3, 0.01, 5, 1.6);
	//Ptr<xfeatures2d::SURF> sift1 = xfeatures2d::SURF::create(10,4,3,true);//
	//sift1->detectAndCompute(image_1, Mat(), keypoints_1, descriptors_1);
	sift1->detect(image_1, keypoints_1);
	//mser->compute(image_1, keypoints_1, descriptors_1);//计算描述矩阵，保存在description中
	des->compute(image_1, keypoints_1, descriptors_1);
	Mat output1;
	drawKeypoints(image_1, keypoints_1, output1);


	//待配准图像特征点检测与描述
	Mat descriptors_2;
	vector<KeyPoint> keypoints_2;
	vector<Mat> sar_harris_fun_2, amplit_2, orient_2;
	double detect2_beg = (double)getTickCount();

	Ptr<FastFeatureDetector> fast2 = FastFeatureDetector::create(10,true);
	Ptr<MSER> mser2 = MSER::create(2, 10, 5000, 0.5, 0.3);
	//Ptr<ORB> orb2 = ORB::create(500, pow(2, 1.0 / 3.0), 2, 10, 0, 2, ORB::HARRIS_SCORE, 10, 10);
	Ptr<xfeatures2d::SIFT> sift2 = xfeatures2d::SIFT::create(500, 3, 0.01, 2, 1.6);//
	//Ptr<xfeatures2d::SURF> sift2 = xfeatures2d::SURF::create(100,4,3, true);//
	//sift2->detectAndCompute(image_2, Mat(), keypoints_2, descriptors_2);
	sift2->detect(image_2, keypoints_2);
	//sift2->compute(image_2, keypoints_2, descriptors_2);//计算描述矩阵，保存在description
	des->compute(image_2, keypoints_2, descriptors_2);
	Mat output2;
	drawKeypoints(image_2, keypoints_2, output2);
	

	//描述子匹配
	double match_beg = (double)getTickCount();
	vector<vector<DMatch>> dmatchs;
	vector<DMatch> dmatchs1;
	vector<vector<DMatch>> dmatchs2;
	if (descriptors_1.type() != CV_32F) descriptors_1.convertTo(descriptors_1, CV_32F);
	if (descriptors_2.type() != CV_32F) descriptors_2.convertTo(descriptors_2, CV_32F);
	match_des(descriptors_1, descriptors_2, dmatchs, COS);
	Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher();
	//Ptr<DescriptorMatcher> matcher = new BFMatcher(2);
	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	//matcher->match(descriptors_1, descriptors_2, dmatchs1);
	matcher->knnMatch(descriptors_1, descriptors_2, dmatchs2, 2);
	
	Mat image_match1, image_match2;
	drawMatches(image_1, keypoints_1, image_2, keypoints_2, dmatchs2, image_match1);
	drawMatches(image_1, keypoints_1, image_2, keypoints_2, dmatchs, image_match2);

	vector<DMatch> right_matchs;
	Mat matched_line;
	Mat homography = match(image_1, image_2, dmatchs2, keypoints_1, keypoints_2, change_model, right_matchs, matched_line);



	double total_time = ((double)getTickCount() - total_beg) / getTickFrequency();
	cout << "总花费时间是： " << total_time << endl;

	//图像融合
	Mat fusion_image, mosaic_image, matched_image;
	image_fusion(image_1, image_2, homography, fusion_image, mosaic_image);
	return matched_line;

}

const string path = "D:/Image-Registration-master/test2/testA";

int main()
{

	int i = 0;
	Mat reference = imread(path + "/1.png");
	Mat toBeAlign = imread(path + "/16.png");
	Mat a = imageRegistration(reference, toBeAlign);
	/*_finddata_t file_info;
	string current_path = path + "/*.png";
	intptr_t handle = _findfirst(current_path.c_str(), &file_info);
	if (handle == -1) return 1;
	do
	{
		Mat toBeAlign = imread(path + "/" + file_info.name);
		Mat Align = imageRegistration(reference, toBeAlign);
		imwrite("D:/Image-Registration-master/test2/testP4/Align2/Align" + to_string(i) + ".jpg", Align);
		i++;

	} while (!_findnext(handle, &file_info));

	_findclose(handle);*/

	//getchar();
	return 0;
	
}