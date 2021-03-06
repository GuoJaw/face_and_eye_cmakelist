

#ifndef _SSD_DETECT_HPP_
#define _SSD_DETECT_HPP_

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include "highgui.h"  
#include "cxcore.h"
#include "cv.h" 


using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

class Detector {
    public:
        Detector();
        ~Detector();
        void Set(string model_file,  string weights_file,  string mean_file, string mean_value,  const int isMobilenet);
	
        std::vector<vector<float> > Detect(const cv::Mat& img);
	
        void Postprocess_Face(cv::Mat& img, const float confidence_threshold, std::vector<vector<float> > detections,cv::Mat& FaceImg);
	void Postprocess_Eye( cv::Mat& Faceimg, const float confidence_threshold, std::vector<vector<float> > detections,std::vector<cv::Mat>& EyeVec);
    private:
        void SetMean(const string& mean_file, const string& mean_value);
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);   
        std::string flt2str(float f);    
        std::string int2str(int n) ;

    private:
        caffe::shared_ptr<Net<float> > net_;   //std::shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        float scale_;
        
        std::string CLASSES_FACE[2] = { "background", "face" };
        std::string CLASSES_EYE[2] = { "background", "eye" };
};
#endif


