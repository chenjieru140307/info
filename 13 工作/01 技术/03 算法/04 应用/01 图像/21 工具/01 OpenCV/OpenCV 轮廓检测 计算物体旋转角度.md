

# 可以补充进来的

- 感觉这个还是很厉害的，是怎么判断这个的旋转角度的？
- 还是要自己跑一遍的。

# 【OpenCV】OpenCV轮廓检测，计算物体旋转角度

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/181031/6k4hl17Am2.png?imageslim">
</p>


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/181031/5K7E46Gg34.png?imageslim">
</p>

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/181031/5KFbdF9hji.png?imageslim">
</p>


OpenCV轮廓检测，计算物体旋转角度

效果还是有点问题的，希望大家共同探讨一下


```python
// FindRotation-angle.cpp : 定义控制台应用程序的入口点。
//

// findContours.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"



#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#pragma comment(lib,"opencv_core2410d.lib")      
#pragma comment(lib,"opencv_highgui2410d.lib")      
#pragma comment(lib,"opencv_imgproc2410d.lib")

#define PI 3.1415926

using namespace std;
using namespace cv;



int hough_line(Mat src)
{
 //【1】载入原始图和 Mat 变量定义  
 Mat srcImage = src;//imread("1.jpg");  //工程目录下应该有一张名为 1.jpg的素材图
 Mat midImage,dstImage;//临时变量和目标图的定义

 //【2】进行边缘检测和转化为灰度图
 Canny(srcImage, midImage, 50, 200, 3);//进行一此 canny 边缘检测
 cvtColor(midImage,dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图

 //【3】进行霍夫线变换
 vector<Vec4i> lines;//定义一个矢量结构 lines 用于存放得到的线段矢量集合
 HoughLinesP(midImage, lines, 1, CV_PI/180, 80, 50, 10 );

 //【4】依次在图中绘制出每条线段
 for( size_t i = 0; i < lines.size(); i++ )
 {
  Vec4i l = lines[i];
  line( dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186,88,255), 1, CV_AA);
 }

 //【5】显示原始图  
 imshow("【原始图】", srcImage);

 //【6】边缘检测后的图
 imshow("【边缘检测后的图】", midImage);

 //【7】显示效果图  
 imshow("【效果图】", dstImage);

 //waitKey(0);

 return 0;  
}

int main()
{
 // Read input binary image

 char *image_name = "test.jpg";
 cv::Mat image = cv::imread(image_name,0);
 if (!image.data)
  return 0;

 cv::namedWindow("Binary Image");
 cv::imshow("Binary Image",image);



 // 从文件中加载原图  
​    IplImage *pSrcImage = cvLoadImage(image_name, CV_LOAD_IMAGE_UNCHANGED);  

    // 转为 2 值图

  cvThreshold(pSrcImage,pSrcImage,200,255,cv::THRESH_BINARY_INV);
​    

    image = cv::Mat(pSrcImage,true);

    cv::imwrite("binary.jpg",image);

 // Get the contours of the connected components
 std::vector<std::vector<cv::Point>> contours;
 cv::findContours(image,
  contours, // a vector of contours
  CV_RETR_EXTERNAL, // retrieve the external contours
  CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

 // Print contours' length
 std::cout << "Contours: " << contours.size() << std::endl;
 std::vector<std::vector<cv::Point>>::const_iterator itContours= contours.begin();
 for ( ; itContours!=contours.end(); ++itContours)
 {

  std::cout << "Size: " << itContours->size() << std::endl;
 }

 // draw black contours on white image
 cv::Mat result(image.size(),CV_8U,cv::Scalar(255));
 cv::drawContours(result,contours,
  -1, // draw all contours
  cv::Scalar(0), // in black
  2); // with a thickness of 2

 cv::namedWindow("Contours");
 cv::imshow("Contours",result);






 // Eliminate too short or too long contours
 int cmin= 100;  // minimum contour length
 int cmax= 1000; // maximum contour length
 std::vector<std::vector<cv::Point>>::const_iterator itc= contours.begin();
 while (itc!=contours.end()) {

  if (itc->size() < cmin || itc->size() > cmax)
   itc= contours.erase(itc);
  else
   ++itc;
 }

 // draw contours on the original image
 cv::Mat original= cv::imread(image_name);
 cv::drawContours(original,contours,
  -1, // draw all contours
  cv::Scalar(255,255,0), // in white
  2); // with a thickness of 2

 cv::namedWindow("Contours on original");
 cv::imshow("Contours on original",original);



 // Let's now draw black contours on white image
 result.setTo(cv::Scalar(255));
 cv::drawContours(result,contours,
  -1, // draw all contours
  cv::Scalar(0), // in black
  1); // with a thickness of 1
 image= cv::imread("binary.jpg",0);

 //imshow("lll",result);
 //waitKey(0);

 // testing the bounding box
 //////////////////////////////////////////////////////////////////////////////
 //霍夫变换进行直线检测，此处使用的是 probabilistic Hough transform（cv::HoughLinesP）而不是 standard Hough transform（cv::HoughLines）

 cv::Mat result_line(image.size(),CV_8U,cv::Scalar(255));
 result_line = result.clone();

 hough_line(result_line);

 //Mat tempimage;

 //【2】进行边缘检测和转化为灰度图
 //Canny(result_line, tempimage, 50, 200, 3);//进行一此 canny 边缘检测
 //imshow("canny",tempimage);
 //waitKey(0);

 //cvtColor(tempimage,result_line, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
 vector<Vec4i> lines;

 cv::HoughLinesP(result_line,lines,1,CV_PI/180,80,50,10);

 for(int i = 0; i < lines.size(); i++)
 {
  line(result_line,cv::Point(lines[i][0],lines[i][1]),cv::Point(lines[i][2],lines[i][3]),Scalar(0,0,0),2,8,0);
 }
 cv::namedWindow("line");
 cv::imshow("line",result_line);
 //waitKey(0);

 /////////////////////////////////////////////////////////////////////////////////////////////
 //

 //std::vector<std::vector<cv::Point>>::const_iterator itc_rec= contours.begin();
 //while (itc_rec!=contours.end())
 //{
 // cv::Rect r0= cv::boundingRect(cv::Mat(*(itc_rec)));
 // cv::rectangle(result,r0,cv::Scalar(0),2);
 //  ++itc_rec;
 //}



 //cv::namedWindow("Some Shape descriptors");
 //cv::imshow("Some Shape descriptors",result);


 CvBox2D    End_Rage2D;
 CvPoint2D32f rectpoint[4];
 CvMemStorage *storage = cvCreateMemStorage(0);  //开辟内存空间


 CvSeq*      contour = NULL;    //CvSeq类型 存放检测到的图像轮廓边缘所有的像素值，坐标值特征的结构体以链表形式

 cvFindContours( pSrcImage, storage, &contour, sizeof(CvContour),CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);//这函数可选参数还有不少



 for(; contour; contour = contour->h_next)  //如果 contour 不为空，表示找到一个以上轮廓，这样写法只显示一个轮廓
  //如改为 for(; contour; contour = contour->h_next) 就可以同时显示多个轮廓
 {

  End_Rage2D = cvMinAreaRect2(contour);  
  //代入 cvMinAreaRect2 这个函数得到最小包围矩形  这里已得出被测物体的角度，宽度，高度，和中点坐标点存放在 CvBox2D 类型的结构体中，
  //主要工作基本结束。
  for(int i = 0;i< 4;i++)
  {
​    //CvArr* s=(CvArr*)&result;
   //cvLine(s,cvPointFrom32f(rectpoint[i]),cvPointFrom32f(rectpoint[(i+1)%4]),CV_G(0,0,255),2);
   line(result,cvPointFrom32f(rectpoint[i]),cvPointFrom32f(rectpoint[(i+1)%4]),Scalar(125),2);
  }
  cvBoxPoints(End_Rage2D,rectpoint);

 std::cout <<" angle:\n"<<(float)End_Rage2D.angle << std::endl;      //被测物体旋转角度

 }
 cv::imshow("lalalal",result);
 cv::waitKey();
 return 0;


}
```



<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/181031/GgC8Elib7K.png?imageslim">
</p>




这个是原来实现的代码的博客文章：http://www.linuxidc.com/Linux/2015-02/114135.htm




# 相关

- [【OpenCV】OpenCV轮廓检测，计算物体旋转角度](https://blog.csdn.net/Taily_Duan/article/details/54964380?utm_source=blogkpcl14)
