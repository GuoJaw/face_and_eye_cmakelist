#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
 
using namespace caffe;  // NOLINT(build/namespaces)
//using namespace cv;
//using namespace std;
 
class Detector {
public:
    Detector(const string& model_file,
        const string& weights_file,
        const string& mean_file,
        const string& mean_value);
 
    std::vector<vector<float> > Detect(const cv::Mat& img);
 
private:
    void SetMean(const string& mean_file, const string& mean_value);
 
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
 
    void Preprocess(const cv::Mat& img,
        std::vector<cv::Mat>* input_channels);
 
private:
    caffe::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};
 
Detector::Detector(const string& model_file,
    const string& weights_file,
    const string& mean_file,
    const string& mean_value) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif
 
    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);
 
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
 
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
 
    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);
}
 
std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
        input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
 
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
 
    Preprocess(img, &input_channels);
 
    net_->Forward();
 
    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1) {
            // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    return detections;
}
 
/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
            "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
 
        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";
 
        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }
 
        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);
 
        /* Compute the global mean pixel value and create a mean image
        * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) <<
            "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
            "Specify either 1 mean_value or as many as channels: " << num_channels_;
 
        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}
 
 
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];
 
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}
 
void Detector::Preprocess(const cv::Mat& img,
    std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;
 
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;
 
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
 
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);
 
    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
 
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}
 
 
 
 
DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");
 
const string& mean_file = FLAGS_mean_file;
const string& mean_value = FLAGS_mean_value;
 
/*------------------------------------------------------------------------------*/
 
int eye_i = 988, frame_eye = 0;



string num2label_face(int num)
{
    string labels[2]{" ", "face"};
    return labels[num];
}
string num2label_eye(int num)
{
    string labels[2]{" ", "eye"};
    return labels[num];
}
 
void ssdDetect_eye(Detector& detector, cv::Mat& imgFace)
{
    if (!imgFace.data) //如果没有检测到脸
        return;
 
    std::vector<vector<float> > detections = detector.Detect(imgFace);
 
    /* Print the detection results. */
    for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= 0.6)  // 
        {
            int posx = static_cast<int>(d[3] * imgFace.cols);
            int posy = static_cast<int>(d[4] * imgFace.rows);
            int posw = static_cast<int>(d[5] * imgFace.cols) - posx;
            int posh = static_cast<int>(d[6] * imgFace.rows) - posy;

	    //获取人眼
	    cv::Rect pos(posx, posy, posw, posh);
	    cv::Mat eye_ROI( imgFace, pos);
	    
	    frame_eye++;
	    if(frame_eye % 5 == 0)  //每检测到20次人眼就保存一次
	    {

	      //写入目录
	      	    char buffer_eye[256];
		    sprintf(buffer_eye,"/home/gjw/EyeData/%d.jpg",eye_i++);  //数据集保存路径
		    std::string dir_eye = std::string(buffer_eye);
		    std::cout << dir_eye << std::endl;

		    //cv::Mat grepimg;
		    //cv::cvtColor(eye_ROI, grepimg, CV_RGB2GRAY);  
		    //cv::imwrite(dir_eye, grepimg);
		    cv::imwrite(dir_eye, eye_ROI);
	    }
	    
            char buffer[50];
            sprintf(buffer, "%.2f", score);  //格式化输出
            std::string words = std::string(buffer);
            words = num2label_eye(static_cast<int>(d[1])) + ":" + words;
	    
	    cv::rectangle(imgFace, pos, cv::Scalar(0,0,255));
            cv::putText(imgFace, words, cv::Point(posx, posy), CV_FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0,0,255));    
        }
    }
    cv::imshow("imgFace", imgFace);
}
void ssdDetect_Face(Detector& detector, cv::Mat& img, cv::Mat& imgFace)
{
    std::vector<vector<float> > detections = detector.Detect(img);
 
    /* Print the detection results. */
    for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
	int posx = static_cast<int>(d[3] * img.cols) - 30;
	int posy = static_cast<int>(d[4] * img.rows);
	int posw = static_cast<int>(d[5] * img.cols) - posx + 50;
	int posh = static_cast<int>(d[6] * img.rows) - posy;
        if ((score >= 0.6 )&& (posx>=0 && posy>=0 && posw>=0 && posh>=0)&& ((posx+posw)<img.cols) && ((posy + posh)<img.rows))  // 
        {
            cv::Rect pos(posx, posy, posw, posh);
            cv::rectangle(img, pos, cv::Scalar(255,0,0));
            imgFace = img(pos);
            char buffer[50];
            sprintf(buffer, "%.2f", score);  //格式化输出
            std::string words = std::string(buffer);
            words = num2label_face(static_cast<int>(d[1])) + ":" + words;
            cv::putText(img, words, cv::Point(posx, posy), CV_FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(255,0,0));
        }
    }
}
/*------------------------------------------------------------------------------*/
 
 
//脸
const string& model_file_face("../model/face/deploy_face.prototxt");  //prototxt文件
const string& weights_file_face("../model/face/VGG_Face_SSD_300x300_iter_20000.caffemodel");  //caffemodel文件
 
//睁眼闭眼 
const string& model_file_eye("../model/eye/deploy_eye.prototxt");  //prototxt文件
const string& weights_file_eye("../model/eye/VGG_Eye_SSD_300x300_iter_20000.caffemodel");;  //caffemodel文件
 
int main(int argc, char** argv) 
{
    cv::Mat img, imgFace;
    Detector detector_face(model_file_face, weights_file_face, mean_file, mean_value);
    Detector detector_eye(model_file_eye, weights_file_eye, mean_file, mean_value);
/////
    cv::VideoCapture cap;
/*
    if(atoi(argv[1]) == 0)  //如果输入的值是0，则打开摄像头
        cap.open(0);
    else 
	cap.open(argv[1]);  //否则，打开视频文件
*/

    cap.open(0);  //否则，打开视频文件

    //初始化
    cvNamedWindow("SSD", 0);  //0表示窗口大小可调
    
    double fps = 0, t = 0.0;;
    int frame_count = 0;
    while (true) 
    {
        //求FPS
        t = (double)cv::getTickCount();
        cap.read(img);

	cv::Mat kernel = (cv::Mat_<float>(4,4)<<0,-1,0,0,6,0,0,-1,0);
	cv::filter2D(img,img,CV_8UC3,kernel);	

        //
        char c = cv::waitKey(1);
        if (c == 27)
        {
            if (cap.isOpened())
            {
                cap.release();
            }
            return -1;
        }
         
        ssdDetect_Face(detector_face, img, imgFace);   //SSD检测人脸，将检测到的人脸抠出来，用于人眼检测
    	if (imgFace.data) //如果没有检测到脸
	        ssdDetect_eye(detector_eye, imgFace);    //SSD检测睁眼闭眼
 
        //FPS
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
 
        char str[20]; 
        sprintf(str, "%.2f", fps);
        std::string fpsString("FPS:");
        fpsString += str;
        putText(img, fpsString, cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
        std::cout << "FPS=" << fps << std::endl;
 
        cv::imshow("SSD", img);
        ++frame_count;
    }
}

