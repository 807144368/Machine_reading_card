#include<iostream>
#include<opencv2\opencv.hpp>
#include<vector>
#include<math.h>
#include<algorithm>
#include "opencv2/ml.hpp"
#include "ml.h"
using namespace std;
using namespace cv::ml;
using namespace cv;




bool cmp(vector<cv::Point>countourA, vector<cv::Point>countourB) {
	return contourArea(countourA) > contourArea(countourB);
}
bool cmpRect(Rect a ,Rect b) {
	return a.x < b.x;
}
vector<cv::Point> GetCorner(vector<cv::Point>MaxContours, Point center_point);
Point GetCenter(vector<Point>bounding_point)
{
	Point center_point = { 0,0 };
	if (bounding_point.size() == 0)
		return{ 0,0 };
	for (size_t i = 0; i < bounding_point.size(); i++)
	{
		center_point.x += bounding_point[i].x;
		center_point.y += bounding_point[i].y;
	}
	center_point.x = center_point.x / bounding_point.size();
	center_point.y = center_point.y / bounding_point.size();
	return center_point;
}

double GetDistance(Point2f p1, Point2f p2)
{
	return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

cv::Mat Img_Threshold(Mat gray) {
	Mat mid,result;
	int maxVal = 255;
	int BlackSize = (gray.cols/16)*2+1;
	double C =9;
	blur(gray, gray, Size(3, 3));
	adaptiveThreshold(gray,
		mid,maxVal,
		CV_ADAPTIVE_THRESH_MEAN_C, 
		CV_THRESH_BINARY, 
		BlackSize, C);
	bitwise_not(mid, result);
	return result;
}
Mat toGray(Mat srcImg) {
	Mat grayImg;
	cvtColor(srcImg, grayImg, CV_BGR2GRAY);
	Mat binImg;
	binImg = Img_Threshold(grayImg);
	return binImg;
}
cv::Mat getROI(Mat srcImg,Mat binImg,int order) {
	
	vector<vector<cv::Point>>contours;
	vector<Vec4i> hierarchy;
	findContours(binImg,contours,hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE,
		Point(0, 0));
	vector<Point> contours_poly;
	//vector<Rect> boundRect(contours.size());
	//vector<Point2f>center(contours.size());
	Rect max_rect;
	//int size, max_size=0, sec_size=0,max_at,sec_at;
	sort(contours.begin(), contours.end(), cmp);
	approxPolyDP(Mat(contours[order]), contours_poly, 3, true);//对图像进行多边形拟合
	max_rect = boundingRect(Mat(contours_poly));
	 //得到中心点
	Point center_point = GetCenter(contours[order]);
	//得到四点
	vector<cv::Point>   FourCorner=GetCorner(contours[order], center_point);
	circle(srcImg, FourCorner[0], 3, Scalar(255, 0, 255), 10, 8);
	circle(srcImg, FourCorner[1], 3, Scalar(255, 0, 255), 10, 8);
	circle(srcImg, FourCorner[2], 3, Scalar(255, 0, 255), 10, 8);
	circle(srcImg, FourCorner[3], 3, Scalar(255, 0, 255), 10, 8);
	rectangle(srcImg, max_rect, Scalar(200,33,22));

	Point2f max_point[] = {
		Point2f(FourCorner[0]),
		Point2f(FourCorner[1]),
		Point2f(FourCorner[2]),
		Point2f(FourCorner[3]),
	};
	//目标点
	Size p_size(max_rect.width, max_rect.height);
	Mat img_dst(p_size, CV_8UC3);
	Point2f dst_point[] = {
		Point2f(0,0),
		Point2f(0,max_rect.height),
		Point2f(max_rect.width,max_rect.height),
		Point2f(max_rect.width,0),
	};


	Mat h= getPerspectiveTransform(max_point, dst_point);
	 cv::warpPerspective(srcImg, img_dst, h, p_size, INTER_LINEAR);//透视变化

	imwrite("最大ROI区域.jpg",img_dst);//最大的Roi区域
 
	return img_dst;
}
vector<cv::Point> GetCorner(vector<cv::Point>MaxContours, Point center_point) {

	vector<cv::Point> FourCorner;
	double temp;
	double tl = 0, bl = 0, br = 0, tr = 0;
	Point ptl, pbl, pbr, ptr;
	for (size_t i = 0; i < MaxContours.size(); i++) {
		Point ptemp = MaxContours[i];
		temp = GetDistance(center_point, ptemp);
		if (ptemp.x <= center_point.x && ptemp.y <= center_point.y) {
			if (temp > tl) {//左上角
				tl = temp;
				ptl = ptemp;
			}
		}
		else if (ptemp.x <= center_point.x && ptemp.y > center_point.y) {
			if (temp > bl) {//左下
				bl = temp;
				pbl = ptemp;
			}
		}
		else if (ptemp.x >= center_point.x && ptemp.y >= center_point.y) {
			if (temp > br) {//右下
				br = temp;
				pbr = ptemp;
			}
		}
		else {
			if (temp > tr) {//右上
				tr = temp;
				ptr = ptemp;
			}
		}
	}
	FourCorner.push_back(ptl);
	FourCorner.push_back(pbl);
	FourCorner.push_back(pbr);
	FourCorner.push_back(ptr);
	return FourCorner;
}


void JudgeSelect(Mat roi_img) {
	
	//截取出选择区域
	Rect select(roi_img.cols*0.0153, roi_img.rows*0.3378, roi_img.cols - roi_img.cols*0.0186, roi_img.rows- roi_img.rows*0.6134);//5.jpg 18 375 18 680
	Mat select_img = roi_img(select);
	imwrite("selectImg.jpg", select_img);
	Mat binImg;
	binImg = toGray(select_img);
	//namedWindow("bin", WINDOW_NORMAL);
	//imshow("bin",binImg);
//	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//	Mat resImg;
//	dilate(binImg, resImg, element);//膨胀化
//	erode(resImg, resImg, element);//腐蚀化
	//检测选项轮廓
	vector<vector<cv::Point>>contours;
	vector<Vec4i> hierarchy;
	findContours(binImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE,
		Point(0, 0));	
	
    vector<Rect> BoundRect;	
	double t_area;
	for (size_t i = 0; i < contours.size(); i++) {
		t_area = contourArea(contours[i]);
		if (t_area >180&&t_area<=750) {//筛选出选项并储存  180 750
			Rect rect;
			rect= boundingRect(Mat(contours[i]));
			if (rect.height>=12&& rect.width>=28&&rect.height<=26) { //12 28
				BoundRect.push_back(rect);
                 rectangle(select_img, rect, Scalar(255, 0, 255),CV_FILLED);
			}
			 
		}
	}
	String anwsers[51];
	//得到4*3的单元格
	int ge_cols =select_img.cols / 4; 
	int ge_rows = select_img.rows / 3;
	//遍历所有填涂的答案
	for (int i = 0; i < BoundRect.size(); i++) {
		Rect rect = BoundRect[i];
		char *anwser;
		Point  p;
		int	index = 1 + (rect.x / ge_cols) * 5 + (rect.y / ge_rows) * 20 + (rect.y%ge_rows) / (ge_rows / 5);
		int mmm=(rect.x%ge_cols) / (ge_cols / 4);
		switch ((rect.x%ge_cols) / (ge_cols/ 4)) {
		case 0:anwser = "A"; break;
		case 1:anwser = "B"; break;
		case 2:anwser = "C"; break;
		case 3:anwser = "D"; break;
		
		}
		anwsers[index] += anwser;
		 putText(select_img, anwser, rect.tl(), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2, 8, false);//检验
	}
//	imshow("test", select_img);
	for (int i = 1; i <=50; i++) {
		cout << i  << " " << anwsers[i] << endl;
	}
	imwrite("selcet.jpg", select_img);
}

void SeclectNumber(Mat roi_img) {
  
    Rect num_roi(roi_img.cols*0.0115, roi_img.rows*0.01129, roi_img.cols - 2* roi_img.cols*0.0115, roi_img.rows - roi_img.rows*0.7257);  //0.7457
	Mat num_img = roi_img(num_roi);
	imwrite("numImg.jpg", num_img);
	Mat binImg;
	binImg = toGray(num_img);
	//imwrite("binImg.jpg",binImg);
	vector<vector<cv::Point>>contours_1;//最大行
	vector<vector<cv::Point>>contours_1_18;//最大行身份证号
	vector<Vec4i> hierarchy_1; 
	vector<Vec4i> hierarchy_1_18;
	findContours(binImg, contours_1, hierarchy_1, RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE,
		Point(0, 0));
	sort(contours_1.begin(), contours_1.end(), cmp);
	Rect rect_1 = boundingRect(Mat(contours_1[0]));
	rectangle(num_img, rect_1, Scalar(255, 0, 0), 1, 1, 0);
	//imwrite("num_img.jpg",num_img);
	Mat IdImg = binImg(rect_1);
	//imwrite("Id.jpg", IdImg);
	findContours(IdImg, contours_1_18, hierarchy_1_18, RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE,
		Point(0, 0));
	sort(contours_1_18.begin(), contours_1_18.end(), cmp);
	vector<Rect> boundRect_1_18;
	for (size_t i = 1; i < 19; i++) {
		Rect rect = boundingRect(Mat(contours_1_18[i]));
		//rectangle(IdImg, rect, Scalar(255, 0, 0),1, 1, 0);
		boundRect_1_18.push_back(rect);
	}
	sort(boundRect_1_18.begin(), boundRect_1_18.end(), cmpRect);
	imwrite("BIN.jpg", IdImg);
	for (size_t i = 0; i < boundRect_1_18.size(); i++) {
		//rectangle(num_img, boundRect_1_18[i], Scalar(255, 0, 0), 1, 1, 0);
		Mat midImg =IdImg(boundRect_1_18[i]);
		Mat resImg;
		resize(midImg, resImg, Size(32, 32), 0, INTER_LINEAR);
		Rect rect(2, 2, resImg.cols - 4, resImg.rows - 4);
		Mat Img = resImg(rect);
		cv::String ImgSrc = "./save/" + to_string(i) + ".bmp";
		imwrite(ImgSrc,Img);
	}

}

void trainSVM() {
	Mat classes;
	vector<string> img_path;//输入文件名变量     
	vector<int> img_catg;
	int nLine = 0;
	string buf;

	//1.读入训练数据
	ifstream svm_data("./train_img/img.txt");//训练样本图片的路径都写在这个txt文件中，使用bat批处理文件可以得到这个txt文件       
	unsigned long n;
	while (svm_data)//将训练样本文件依次读取进来      
	{
		if (getline(svm_data, buf))
		{
			nLine++;
			if (nLine % 2 == 0)//注：奇数行是图片全路径，偶数行是标签   
			{
				img_catg.push_back(atoi(buf.c_str()));//atoi将字符串转换成整型，标志(0,1，2，...，9)，注意这里至少要有两个类别，否则会出错      
			}
			else
			{
				img_path.push_back(buf);//图像路径      
			}
		}
	}
	svm_data.close();//关闭文件      

	Mat data_mat, labels_mat;
	int  nImgNum = nLine / 2; //nImgNum是样本数量，只有文本行数的一半，另一半是标签    
	cout << " 共有样本个数为： " << nImgNum << endl;
	//data_mat为所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
	data_mat = Mat::zeros(nImgNum, 324, CV_32FC1);  //行、列、类型；第二个参数，即矩阵的列是由下面的descriptors的大小决定的，可以由descriptors.size()得到，且对于不同大小的输入训练图片，这个值是不同的    
													//类型矩阵,存储每个样本的类型标志      
													//labels_mat为训练样本的类别向量，行数等于所有样本的个数，列数等于1；暂时，后面会修改，比如样本0，就为0，样本1就为1
	labels_mat = Mat::zeros(nImgNum, 1, CV_32SC1);
	Mat src;
	Mat trainImg = Mat(Size(28, 28), CV_8UC3);//需要分析的图片，这里默认设定图片是28*28大小，所以上面定义了324，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行      

	  //2.处理HOG特征  
	for (string::size_type i = 0; i != img_path.size(); i++)
	{
		src = imread(img_path[i].c_str());
		if (src.empty())
		{
			cout << " can not load the image: " << img_path[i].c_str() << endl;
			continue;
		}

		cout << " 处理： " << img_path[i].c_str() << endl;

		resize(src, trainImg, trainImg.size());

		HOGDescriptor *hog = new HOGDescriptor(Size(28, 28), Size(14, 14), Size(7, 7), Size(7, 7), 9);
		vector<float>descriptors;//存放结果    为HOG描述子向量    
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //Hog特征计算，检测窗口移动步长(1,1)     

		cout << "HOG描述子向量个数: " << descriptors.size() << endl;

		n = 0;
		int    number = descriptors.size();
		//将计算好的HOG描述子复制到样本特征矩阵data_mat  
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++) {
			data_mat.at<float>(i, n) = *iter;
			n++;
		}
		labels_mat.at<int>(i, 0) = img_catg[i];
		cout << " 处理完毕: " << img_path[i].c_str() << " " << img_catg[i] << endl;
	}

	Mat(labels_mat).copyTo(classes);


	//3.创建SVM分类器并设置参数
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);  //核函数，后期重点分析的地方  SVM::RBF为径向基（RBF）核函数（高斯核函数）

	svm->setDegree(10.0);
	svm->setGamma(0.09);
	svm->setCoef0(1.0);
	svm->setC(10.0);
	svm->setNu(0.5);
	svm->setP(1.0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));

	
	cout << "开始训练..." << endl;
	Ptr<TrainData> traindata = ml::TrainData::create(data_mat, ROW_SAMPLE, classes);
	// 训练分类器
	svm->train(traindata);
	cout << "开始识别......" << endl;
	svm->save("svm.xml");
	cout << "训练好了！！！" << endl;
	
}
void ViewNumber(int k) {
	Ptr<ml::SVM> svm = ml::SVM::load("svm.xml");
	int answer[2][18] = { 5,1,0,1,8,9,5,9,1,7,0,3,0,4,2,6,4,10, 
	                                  5,1,0,8,9,2,1,9,9,7,0,4,2,3,6,4,5,10 };
	int cnt = 0;
	Mat image, imageCV_8UC3, imageCV_32FC1;
	for (int i = 0; i < 18; i++) {
		string imagePath = "./save/" + to_string(i) + ".bmp";
		image = imread(imagePath);
		image.convertTo(imageCV_8UC3, CV_8UC3);

		HOGDescriptor *hog = new HOGDescriptor(cvSize(28, 28), cvSize(14, 14), cvSize(7, 7), cvSize(7, 7), 9);
		vector<float>descriptors;
		hog->compute(imageCV_8UC3, descriptors, Size(1, 1), Size(0, 0));//Hog特征计算

		imageCV_32FC1 = Mat::zeros(1, 324, CV_32FC1);
		int n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++) {
			imageCV_32FC1.at<float>(0, n) = *iter;
			n++;
		}
		int result = (int)svm->predict(imageCV_32FC1);

		if (result == answer[k][i]) {
			if (result == 10) {
				cout << "\n图片地址：" << imagePath << "，识别结果：" << "x"<< " -----匹配成功！" << endl;
			}
			else {
	             cout << "\n图片地址：" << imagePath << "，识别结果：" << result << " -----匹配成功！" << endl;
			}
		
			cnt++;
		}
		else {
			cout << "\n图片地址：" << imagePath << "，识别结果：" << result << " -----匹配失败！" << endl;
		}
	}
	cout << "处理完成！" << endl;
	cout << "一共识别18个图像，识别准确有" << cnt << "个！" << endl;
	double right = cnt / 18.0;
	cout << "识别准确率为" << right << endl;
	

}
int main()
 {
	
	Mat src_Img,roi_Img;
	src_Img = imread("0.jpg");
	resize(src_Img, src_Img, Size(src_Img.cols / 2, src_Img.rows / 2), 0, INTER_LINEAR);
	Mat bin_Img=toGray(src_Img);
	imwrite("binImg.jpg", bin_Img);
	roi_Img = getROI(src_Img, bin_Img, 0);//找到ROI区

    JudgeSelect( roi_Img);// 选择题答案判断
	SeclectNumber(roi_Img); //数字提取
	ifstream svm ("svm.xml");
	if (!svm) {
		trainSVM();
	}
	ViewNumber(0);//0-1
	waitKey(0);
	int n;
	cin >> n;
	return 0;
}