#include"Sar_sift.h"
#include"match.h"

#include<opencv2\highgui\highgui.hpp>
#include<opencv2\features2d\features2d.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2/xfeatures2d.hpp>

#include<sstream>
#include<vector>
#include<fstream>
#include<io.h>
Mat imageRegistration(Mat& image_1, Mat& image_2, string change_model = string("affine"))
{

	if (!image_1.data || !image_2.data){
		cout << "图像数据加载失败！" << endl;
		return Mat();
	}

	double total_beg = (double)getTickCount();//总时间开始
	//构建Sar_sift对象
	int nums_1 = image_1.rows*image_1.cols;
	int nums_2 = image_2.rows*image_2.cols;
	int nFeatures_1 = cvRound((double)nums_1*0.008);
	int nFeatures_2 = cvRound((double)nums_2*0.008);
	Sar_sift sar_sift_1(nFeatures_1, 8, 2, pow(2, 1.0 / 3.0), 0.00001, 0.04); //限制空间金字塔层数则可以减少运行时间
	Sar_sift sar_sift_2(nFeatures_2, 8, 2, pow(2, 1.0 / 3.0), 0.00001, 0.04);

	Ptr<xfeatures2d::BriefDescriptorExtractor> des = xfeatures2d::BriefDescriptorExtractor::create();
	//参考图像特征点检测与描述
	vector<KeyPoint> keypoints_1;
	vector<Mat> sar_harris_fun_1, amplit_1, orient_1;
	double detect1_beg = (double)getTickCount();
	sar_sift_1.detect_keys(image_1, keypoints_1, sar_harris_fun_1, amplit_1, orient_1);
	double detect1_time = ((double)getTickCount() - detect1_beg) / getTickFrequency();
	cout << "参考图像特征点检测花费时间是： " << detect1_time << "s" << endl;
	cout << "参考图像检测特征点个数是： " << keypoints_1.size() << endl;
	Mat output1;
	drawKeypoints(image_1, keypoints_1, output1);

	double des1_beg = (double)getTickCount();
	Mat descriptors_1;
	sar_sift_1.comput_des(keypoints_1, amplit_1, orient_1, descriptors_1);
	double des1_time = ((double)getTickCount() - des1_beg) / getTickFrequency();
	cout << "参考图像特征点描述花费时间是： " << des1_time << "s" << endl;

	//待配准图像特征点检测与描述
	vector<KeyPoint> keypoints_2;
	vector<Mat> sar_harris_fun_2, amplit_2, orient_2;
	double detect2_beg = (double)getTickCount();
	sar_sift_2.detect_keys(image_2, keypoints_2, sar_harris_fun_2, amplit_2, orient_2);
	double detect2_time = ((double)getTickCount() - detect2_beg) / getTickFrequency();
	cout << "待配准图像特征点检测花费时间是： " << detect2_time << "s" << endl;
	cout << "待配准图像检测特征点个数是： " << keypoints_2.size() << endl;
	Mat output2;
	drawKeypoints(image_2, keypoints_2, output2);

	double des2_beg = (double)getTickCount();
	Mat descriptors_2;
	sar_sift_2.comput_des(keypoints_2, amplit_2, orient_2, descriptors_2);
	double des2_time = ((double)getTickCount() - des2_beg) / getTickFrequency();
	cout << "待配准图像特征点描述花费时间是： " << des2_time << "s" << endl;

	


	//描述子匹配
	double match_beg = (double)getTickCount();
	
	vector<vector<DMatch>> dmatchs;
	vector<vector<DMatch>> dmatchs1;

	Mat image_match1;
	Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher();
	matcher->knnMatch(descriptors_1, descriptors_2, dmatchs, 2);
	drawMatches(image_1, keypoints_1, image_2, keypoints_2, dmatchs, image_match1);


	Mat initial_matched;
	match_des(descriptors_1, descriptors_2, dmatchs1, COS);
	//matcher->knnMatch(descriptors_1, descriptors_2, dmatchs,2);
	drawMatches(image_1, keypoints_1, image_2, keypoints_2, dmatchs1,initial_matched);

	vector<DMatch> right_matchs;
	Mat matched_line;
	Mat homography = match(image_1, image_2, dmatchs1, keypoints_1, keypoints_2, change_model, right_matchs, matched_line);

	/*double match_time = ((double)getTickCount() - match_beg) / getTickFrequency();
	cout << "特征点匹配阶段花费时间是： " << match_time << "s" << endl;
	cout << "待配准图像到参考图像的" << change_model << "变换矩阵是：" << endl;
	cout << homography << endl;

	double total_time = ((double)getTickCount() - total_beg) / getTickFrequency();
	cout << "总花费时间是： " << total_time << endl;*/

	
	//图像融合
	Mat fusion_image, mosaic_image, matched_image;
	Mat result = image_fusion(image_1, image_2, homography, fusion_image, mosaic_image);
	return matched_line;
}

const string path = "D:/Image-Registration-master/test2/testA";

int main()
{
	int i = 0;
	Mat reference = imread(path + "/1.png");
	Mat toBeAlign = imread(path + "/16.png");
	imageRegistration(reference, toBeAlign);
	/*_finddata_t file_info;
	string current_path = path + "/*.png";
	intptr_t handle = _findfirst(current_path.c_str(), &file_info);
	if (handle == -1) return 1;
	do
	{
		Mat toBeAlign = imread(path + "/" + file_info.name);
		Mat Align = imageRegistration(reference, toBeAlign);
		imwrite("D:/Image-Registration-master/test2/testB0/Align/Align" + to_string(i) + ".jpg", Align);
		i++;

	} while (!_findnext(handle, &file_info));

	_findclose(handle);*/

	//getchar();
	return 0;
}