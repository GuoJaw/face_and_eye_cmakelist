
#include "ssd_detect.hpp"
#include "classifer.hpp"

#include <sstream>
#include <iostream>

const int isMobilenet = 1;

DEFINE_string(mean_file, "", "The mean file used to subtract from the input image.");


#if isMobilenet
DEFINE_string(mean_value, "127", "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
#else
DEFINE_string(mean_value, "104,117,123", "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
#endif

DEFINE_string(file_type, "image", "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "result/out.txt", "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.7, "Only store detections with score higher than the threshold.");


string mean_file = FLAGS_mean_file;
string mean_value = FLAGS_mean_value;
string file_type = FLAGS_file_type;
string out_file = FLAGS_out_file;
float confidence_threshold = FLAGS_confidence_threshold;


int main(int argc, char** argv) 
{
    std::cout << "input ssd_main ..."  << std::endl;
    //人脸检测模型
    string model_file_face = "../model/face/MobileNetSSD_deploy.prototxt";  //prototxt
    string weights_file_face = "../model/face/MobileNetSSD_deploy.caffemodel";  //caffemodel

    //人眼检测模型
    string model_file_eye = "../model/eye/MobileNetSSD_deploy.prototxt";  //prototxt
    string weights_file_eye = "../model/eye/MobileNetSSD_deploy.caffemodel";  //caffemodel

    //二分类模型
    string model_file_classifer   = "../model/bvlc_alexnet/deploy.prototxt";  //prototxt
    string trained_file_classifer = "../model/bvlc_alexnet/alexnet_iter_1500.caffemodel";  //caffemodel
    string mean_file_classifer    = "../model/bvlc_alexnet/mean.binaryproto";  //均值文件
    string label_file_classifer   = "../model/bvlc_alexnet/labels.txt";  //标签文件：labels.txt

    //
    std::cout << " confidence_threshold=" << confidence_threshold << std::endl;

    // 加载网络
    Detector detector_face;
    detector_face.Set(model_file_face, weights_file_face, mean_file, mean_value, isMobilenet);
    // 加载网络
    Detector detector_eye;
    detector_eye.Set(model_file_eye, weights_file_eye, mean_file, mean_value, isMobilenet);
    // 加载网络
    Classifier classifier(model_file_classifer, trained_file_classifer, mean_file_classifer, label_file_classifer);

    //加载视频文件
    //cv::VideoCapture cap("../test3.mp4");
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
      std::cout << "Failed to open video: " << std::endl;
    }

    cv::Mat img;
    int frame_count = 0;

    double fps;
    char string[10];
    double t = 0.0;

    cv::namedWindow("res",0);

    while (true)
    {
      bool success = cap.read(img);

      //图像增强:4
      cv::Mat kernel = (cv::Mat_<float>(3,3)<<0,-1,0,0,4,0,0,-1,0);
      cv::filter2D(img,img,CV_8UC3,kernel);

      if (!success)
      {
        std::cout << "Process " << frame_count << std::endl;
        break;
      }

      CHECK(!img.empty()) << "Error when read frame";

      //FPS：开始时间
      t = (double)cv::getTickCount();

      //SSD人脸检测
      std::vector<vector<float> > detections_face = detector_face.Detect(img);

      cv::Mat FaceImg;
      detector_face.Postprocess_Face(img, confidence_threshold, detections_face,FaceImg);   //SSD检测

      if (!FaceImg.empty()) //如果检测到脸
      {
                      std::vector<vector<float> > detections_eye = detector_eye.Detect(FaceImg);

                      std::vector<cv::Mat> EyeVec;
                      detector_eye.Postprocess_Eye(FaceImg, confidence_threshold, detections_eye,EyeVec);   //SSD检测

                      //遍历EyeVec容器，进行二分类
                      int eye_num = 1;
                      for(std::vector<cv::Mat>::iterator iter = EyeVec.begin();iter!=EyeVec.end();iter++)   //逐个判断两只眼的“睁闭状态”
                      {
                                   //眼睛标号
                                   char eye_num_buffer[256] ; sprintf(eye_num_buffer,"%d",eye_num); eye_num++;
                                   //std::string eye_num_str = std::string(eye_num_buffer);
                                   //cv::imshow(eye_num_str, *iter);  //显示眼睛窗口

                                   ///进行二分类
                                   std::vector<Prediction> predictions = classifier.Classify(*iter);  //classifier.Classify(img);
                                   //二分类：获取类别概率较大的
                                   std::string resstr;


                                   //if(predictions[0].second >= predictions[1].second) //如果较大
                                   {
                                               if(predictions[0].first == "open")
                                               {
                                                   char buffer[256];
                                                   sprintf(buffer, "%.2f", predictions[0].second);
                                                   resstr = predictions[0].first + " : " + std::string(buffer);
                                                   std::cout << resstr << std::endl;
                                                   cv::putText(*iter, resstr, cv::Point(2,10), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0), 1, 2);   //将分类结果画到图上
                                               }
                                               else if(predictions[0].first == "close")
                                               {
                                                   char buffer[256];
                                                   sprintf(buffer, "%.2f", predictions[0].second);
                                                   resstr = predictions[0].first + " : " + std::string(buffer);
                                                   std::cout << resstr << std::endl;
                                                   cv::putText(*iter, resstr, cv::Point(2,10), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,255), 1, 2);   //将分类结果画到图上
                                               }
                                       }




                      } //for

                     EyeVec.clear(); //清空容器
                     //cv::imshow("FaceImg",FaceImg);

      }  //人脸

      //FPS
      t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
      fps = 1.0 / t;
      sprintf(string, "%.2f", fps);
      std::string fpsString("FPS:");
      fpsString += string;
      putText(img, fpsString, cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2, 10);
      std::cout << "FPS=" << fps << std::endl;

      //显示图像
      cv::imshow("res",img);
      if((char)cv::waitKey(1) == 'q')
          break;

      ++frame_count;
    }
    /////

    if (cap.isOpened()) {
    cap.release();
    }
      return 0;
}


