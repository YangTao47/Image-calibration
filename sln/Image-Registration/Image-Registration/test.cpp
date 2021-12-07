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
#include<chrono>

using namespace cv;
using namespace std;

int descriptorRange = 20;// 15 OR 30

static void meshgrid(const Range &x_range, const Range &y_range, Mat &X, Mat &Y)
{
	int x_start = x_range.start, x_end = x_range.end;
	int y_start = y_range.start, y_end = y_range.end;
	int width = x_end - x_start + 1;
	int height = y_end - y_start + 1;

	X.create(height, width, CV_32FC1);
	Y.create(height, width, CV_32FC1);

	for (int i = y_start; i <= y_end; i++)
	{
		float *ptr_1 = X.ptr<float>(i - y_start);
		for (int j = x_start; j <= x_end; ++j)
			ptr_1[j - x_start] = j * 1.0f;
	}

	for (int i = y_start; i <= y_end; i++)
	{
		float *ptr_2 = Y.ptr<float>(i - y_start);
		for (int j = x_start; j <= x_end; ++j)
			ptr_2[j - x_start] = i * 1.0f;
	}
}

static void calc_gloh_descriptor(const Mat &amplit, const Mat &orient, Point2f pt, float scale, int d, int n, float *ptr_des)
{
	Point point(cvRound(pt.x), cvRound(pt.y));

	int num_rows = amplit.rows;
	int num_cols = amplit.cols;
	int radius = cvRound(descriptorRange * scale);
	radius = min(radius, min(num_rows / 2, num_cols / 2));//特征点邻域半径

	int radius_x_left = point.x - radius;
	int radius_x_right = point.x + radius;
	int radius_y_up = point.y - radius;
	int radius_y_down = point.y + radius;

	//防止越界
	if (radius_x_left < 0)
		radius_x_left = 0;
	if (radius_x_right > num_cols - 1)
		radius_x_right = num_cols - 1;
	if (radius_y_up < 0)
		radius_y_up = 0;
	if (radius_y_down > num_rows - 1)
		radius_y_down = num_rows - 1;

	//此时特征点周围本矩形区域的中心，相对于该矩形
	int center_x = point.x - radius_x_left;
	int center_y = point.y - radius_y_up;

	//特征点周围区域内像素梯度幅度，梯度角度
	Mat sub_amplit = amplit(Range(radius_y_up, radius_y_down + 1), Range(radius_x_left, radius_x_right + 1));
	Mat sub_orient = orient(Range(radius_y_up, radius_y_down + 1), Range(radius_x_left, radius_x_right + 1));


	//以center_x和center_y位中心，对下面矩形区域进行旋转
	Range x_rng(-(point.x - radius_x_left), radius_x_right - point.x);
	Range y_rng(-(point.y - radius_y_up), radius_y_down - point.y);
	Mat X, Y;
	meshgrid(x_rng, y_rng, X, Y);
	Mat c_rot = X ;
	Mat r_rot = Y ;
	Mat GLOH_angle, GLOH_amplit;
	phase(c_rot, r_rot, GLOH_angle, true);//角度在0-360度之间
	GLOH_amplit = c_rot.mul(c_rot) + r_rot.mul(r_rot);//为了加快速度，没有计算开方

	//三个圆半径平方
	float R1_pow = (float)radius*radius;//外圆半径平方
	float R2_pow = pow(radius*SAR_SIFT_GLOH_RATIO_R1_R2, 2.f);//中间圆半径平方
	float R3_pow = pow(radius*SAR_SIFT_GLOH_RATIO_R1_R3, 2.f);//内圆半径平方

	int sub_rows = sub_amplit.rows;
	int sub_cols = sub_amplit.cols;

	//开始构建描述子,在角度方向对描述子进行插值
	int len = (d * 2 + 1)*n;
	AutoBuffer<float> hist(len);
	for (int i = 0; i < len; ++i)//清零
		hist[i] = 0;

	for (int i = 0; i < sub_rows; ++i)
	{
		float *ptr_amplit = sub_amplit.ptr<float>(i);
		float *ptr_orient = sub_orient.ptr<float>(i);
		float *ptr_GLOH_amp = GLOH_amplit.ptr<float>(i);
		float *ptr_GLOH_ang = GLOH_angle.ptr<float>(i);
		for (int j = 0; j < sub_cols; ++j)
		{
			if (((i - center_y)*(i - center_y) + (j - center_x)*(j - center_x)) < radius*radius)
			{
				float pix_amplit = ptr_amplit[j];//该像素的梯度幅度
				float pix_orient = ptr_orient[j];//该像素的梯度方向
				float pix_GLOH_amp = ptr_GLOH_amp[j];//该像素在GLOH网格中的半径位置
				float pix_GLOH_ang = ptr_GLOH_ang[j];//该像素在GLOH网格中的位置方向

				int rbin, cbin, obin;
				rbin = pix_GLOH_amp < R3_pow ? 0 : (pix_GLOH_amp > R2_pow ? 2 : 1);//rbin={0,1,2}
				cbin = cvRound(pix_GLOH_ang*d / 360.f);
				cbin = cbin > d ? cbin - d : (cbin <= 0 ? cbin + d : cbin);//cbin=[1,d]

				obin = cvRound(pix_orient*n / 360.f);
				obin = obin > n ? obin - n : (obin <= 0 ? obin + n : obin);//obin=[1,n]

				if (rbin == 0)//内圆
					hist[obin - 1] += pix_amplit;
				else
				{
					int idx = ((rbin - 1)*d + cbin - 1)*n + n + obin - 1;
					hist[idx] += pix_amplit;
				}
			}
		}
	}

	//对描述子进行归一化
	float norm = 0;
	for (int i = 0; i < len; ++i)
	{
		norm = norm + hist[i] * hist[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < len; ++i)
	{
		hist[i] = hist[i] / norm;
	}

	//阈值截断
	for (int i = 0; i < len; ++i)
	{
		hist[i] = min(hist[i], DESCR_MAG_THR);
	}

	//再次归一化
	norm = 0;
	for (int i = 0; i < len; ++i)
	{
		norm = norm + hist[i] * hist[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < len; ++i)
	{
		ptr_des[i] = hist[i] / norm;
	}
}


void calc_descriptors(const Mat &amplit, const Mat &orient, const vector<KeyPoint> &keys, Mat &descriptors)
{
	int d = SAR_SIFT_GLOH_ANG_GRID;//d=4或者d=8
	int n = SAR_SIFT_DES_ANG_BINS;//n=8默认

	int num_keys = (int)keys.size();
	int grids = 2 * d + 1;
	descriptors.create(num_keys, grids*n, CV_32FC1);

	for (int i = 0; i < num_keys; ++i)
	{
		Point2f key = keys[i].pt;
		float *ptr_des = descriptors.ptr<float>(i);
		float scale = 1;//特征点所在层的尺度

		//计算该特征点的特征描述子
		calc_gloh_descriptor(amplit, orient, key, scale, d, n, ptr_des);
	}
}


void extractKeyPoints(Mat a, Mat b, vector<KeyPoint>& KeyPoint1, vector<KeyPoint>& KeyPoint2)
{
	if (a.channels() != 1) cvtColor(a, a, CV_BGR2GRAY);
	if (b.channels() != 1) cvtColor(b, b, CV_BGR2GRAY);
	threshold(a, a, 1, 255, THRESH_BINARY_INV + THRESH_OTSU);
	threshold(b, b, 1, 255, THRESH_BINARY_INV + THRESH_OTSU);
	vector<vector<Point>> contours;
	vector<vector<Point>> contours2;
	findContours(a, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	findContours(b, contours2, RETR_LIST, CHAIN_APPROX_SIMPLE);
	
	for (auto&p : contours)
	{
		Rect tmp;
		tmp = boundingRect(p);
		KeyPoint tmp_key;
		tmp_key.pt = (static_cast<Point2f>(tmp.tl()) + static_cast<Point2f>(tmp.br())) / 2;
		KeyPoint1.push_back(tmp_key);
	}
	
	for (auto&p : contours2)
	{
		Rect tmp;
		tmp = boundingRect(p);
		KeyPoint tmp_key;
		tmp_key.pt = (static_cast<Point2f>(tmp.tl()) + static_cast<Point2f>(tmp.br())) / 2;
		KeyPoint2.push_back(tmp_key);
	}

	//  绘制
	/*cvtColor(a, a, CV_GRAY2BGR);
	cvtColor(b, b, CV_GRAY2BGR);

	for (auto&p : KeyPoint1)
	{
		circle(a, p.pt, 5, Scalar(0, 0, 255), 1);
	}
	for (auto&p : KeyPoint2)
	{
		circle(b, p.pt, 5, Scalar(0, 0, 255), 1);
	}*/
	return;
}

Mat match(Mat& reference, Mat& toBeAlign, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<vector<DMatch>>& dmatchs1, string change_model)
{

	vector<Point2f> points1_low;
	vector<Point2f> points2_low;
	vector<Point2f> points1_up;
	vector<Point2f> points2_up;
	vector<bool> inliers;
	vector<DMatch> up_matchs;
	float rmse;
	for (int i = 0; i < dmatchs1.size(); ++i)
	{
		up_matchs.push_back(dmatchs1[i][0]);
		up_matchs.push_back(dmatchs1[i][1]);
	}
	for (size_t i = 0; i < dmatchs1.size(); ++i)
	{
		points1_up.push_back(keypoints_1[dmatchs1[i][0].queryIdx].pt);
		points2_up.push_back(keypoints_2[dmatchs1[i][0].trainIdx].pt);
		points1_up.push_back(keypoints_1[dmatchs1[i][1].queryIdx].pt);
		points2_up.push_back(keypoints_2[dmatchs1[i][1].trainIdx].pt);
	}
	points1_low = points1_up;
	points2_low = points2_up;
	Mat homography = FSC(points1_low, points2_low, points1_up, points2_up, change_model, ransac_error, inliers, rmse);
	//提取出处正确匹配点对
	vector<DMatch> right_matchs;
	auto itt = up_matchs.begin();
	for (auto it = inliers.begin(); it != inliers.end(); ++it, ++itt)
	{
		if (*it)//如果是正确匹配点对
		{
			right_matchs.push_back((*itt));
		}

	}
	//绘制初始连线
	Mat initial_matched;
	drawMatches(reference, keypoints_1, toBeAlign, keypoints_2, dmatchs1, initial_matched);
	//绘制正确匹配点对连线图
	Mat matched_line;
	drawMatches(reference, keypoints_1, toBeAlign, keypoints_2, right_matchs, matched_line, 
		        Scalar(0, 255, 0), Scalar(255, 0, 0),vector<char>(), 2);
	Mat result;
	warpPerspective(toBeAlign, result, homography, Size(reference.cols, reference.rows), 3);
	return matched_line;
}

Mat imageRegistration(Mat& reference, Mat& toBeAlign, string change_model = string("affine"))
{
	resize(toBeAlign, toBeAlign, Size(reference.cols, reference.rows));
	vector<KeyPoint> keypoints_1;
	vector<KeyPoint> keypoints_2;
	extractKeyPoints(reference, toBeAlign, keypoints_1, keypoints_2);

	Sar_sift sar_sift_1(500, 1, 2, pow(2, 1.0 / 3.0), 0.00001, 0.04);
	vector<Mat> harris_fun1, amplit1, orient1;
	vector<Mat> harris_fun2, amplit2, orient2;
	sar_sift_1.build_sar_sift_space(reference, harris_fun1, amplit1, orient1);
	sar_sift_1.build_sar_sift_space(toBeAlign, harris_fun2, amplit2, orient2);
	Mat descriptors, descriptors2;
	calc_descriptors(amplit1[0], orient1[0], keypoints_1, descriptors);
	calc_descriptors(amplit2[0], orient2[0], keypoints_2, descriptors2);

	vector<vector<DMatch>> dmatchs1;
	match_des(descriptors, descriptors2, dmatchs1, COS);

	Mat dst;
	dst = match(reference, toBeAlign, keypoints_1, keypoints_2, dmatchs1, change_model);

	return dst;
}

const string path = "D:/Image-Registration-master/test2/testP4";

int main()
{
	int i = 0;
	Mat reference = imread(path + "/1.png");
	/*Mat toBeAlign = imread(path + "/18.png");
	imageRegistration(reference, toBeAlign);*/
	_finddata_t file_info;
	string current_path = path + "/*.png";
	intptr_t handle = _findfirst(current_path.c_str(), &file_info);
	if (handle == -1) return 1;
	do
	{
		auto start_time = chrono::high_resolution_clock::now();
		Mat toBeAlign = imread(path + "/" + file_info.name);
		Mat Align = imageRegistration(reference, toBeAlign);
		imwrite("D:/Image-Registration-master/test2/testP4/Align3/Align" + to_string(i) + ".jpg", Align);
		auto end_time = chrono::high_resolution_clock::now();
		chrono::milliseconds processing_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
		cout << "算法运行时间" <<processing_time.count() << endl;
		i++;

	} while (!_findnext(handle, &file_info));

	_findclose(handle);
	return 0;
}