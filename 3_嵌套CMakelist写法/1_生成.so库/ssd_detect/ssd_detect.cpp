
#include "ssd_detect.hpp"


Detector::Detector() {}     
Detector::~Detector() {}       

void Detector::Set(string model_file,  string weights_file,  string mean_file, string mean_value, const int isMobilenet)
{
                   
    //#ifdef CPU_ONLY
    //Caffe::set_mode(Caffe::CPU);
    //#else
    Caffe::set_mode(Caffe::GPU);
    //#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file, mean_value);

    if (isMobilenet)
        scale_ = 0.007843;    
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    
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

    /*for (int idx=0; idx<detections.size(); idx++) {
        for (int jdx=0; jdx<detections[idx].size(); jdx++){
            std::cout << "Detect idx=" << idx << " jdx=" << jdx << " detection=" << detections[idx][jdx]<<std::endl;
        }
    }*/
    return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) << "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";

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
        CHECK(mean_file.empty()) << "Cannot specify mean_file and mean_value at the same time";
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
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
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

void Detector::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
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
    if (!mean_.empty()) {
        cv::subtract(sample_float, mean_, sample_normalized);
    } else {
        sample_normalized = sample_float;
    }
    
    if (scale_ > 0) 
        sample_normalized *= scale_;
        
    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
    
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

std::string Detector::flt2str(float f) {  
    ostringstream buffer;
    buffer << f;
    string str = buffer.str();
    return str;
}

std::string Detector::int2str(int n) {  
    std::stringstream ss;
    std::string str;
    ss<<n;
    ss>>str;
    return str;
}


void Detector::Postprocess_Face( cv::Mat& img, const float confidence_threshold, std::vector<vector<float> > detections,cv::Mat& FaceImg) 
{    
    /* Print the detection results. */
    for (int i = 0; i < detections.size(); ++i) 
    {              
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        
        const float score = d[2]; //得分score
        const int p1 =  static_cast<int>(d[3] * img.cols);
        const int p2 =  static_cast<int>(d[4] * img.rows);
        const int p3 =  static_cast<int>(d[5] * img.cols);
        const int p4 =  static_cast<int>(d[6] * img.rows);
	
	//程序一直崩溃，是因为没有做越界判断
        if ((score >= confidence_threshold) && (p1>=0 && p2>=0 && p3>=0 && p4>=0)&& (p1< img.cols && p3 < img.cols && p2 < img.rows && p4 < img.rows))
        {       
            const int label_idx = static_cast<int>(d[1]);  //类别标签     
            const string label= CLASSES_FACE[label_idx];

	    //矩形框
            cv::Point point1 = cv::Point(p1, p2);
            cv::Point point2 = cv::Point(p3, p4);
   
	    //在画之前，将人脸抠出来
	    FaceImg = img(cv::Rect(point1,point2));
	    
	    //显示
            std::string title = label + "/" + flt2str(score);                
            cv::putText(img, title, cv::Point(std::max(p1,15), std::max(p4,15)), cv::FONT_ITALIC, 0.6, cv::Scalar(0, 0, 255), 2);
    	    cv::rectangle(img, point1, point2, cv::Scalar(0, 0, 255), 2);      
        }
    }
}
void Detector::Postprocess_Eye( cv::Mat& Faceimg, const float confidence_threshold, std::vector<vector<float> > detections,std::vector<cv::Mat>& EyeVec) 
{    
    /* Print the detection results. */
    for (int i = 0; i < detections.size(); ++i) 
    {              
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        
        const float score = d[2]; //得分score
        const int p1 =  static_cast<int>(d[3] * Faceimg.cols);
        const int p2 =  static_cast<int>(d[4] * Faceimg.rows);
        const int p3 =  static_cast<int>(d[5] * Faceimg.cols);
        const int p4 =  static_cast<int>(d[6] * Faceimg.rows);
        if ((score >= confidence_threshold) && (p1>=0 && p2>=0 && p3>=0 && p4>=0))
        {       
            const int label_idx = static_cast<int>(d[1]);  //类别标签     
            const string label= CLASSES_EYE[label_idx];

	    //矩形框
            cv::Point point1 = cv::Point(p1, p2);
            cv::Point point2 = cv::Point(p3, p4);
    
            std::string title = label + ": " + flt2str(score);                
            cv::putText(Faceimg, title, cv::Point(point1.x, point2.y+15), cv::FONT_ITALIC, 0.6, cv::Scalar(0, 0, 255), 2);  //画字坐标-->cv::Point(point1.x, point2.y+15),
    	    cv::rectangle(Faceimg, point1, point2, cv::Scalar(0, 0, 255), 2);      
	    
	    //将人脸抠出来
	    cv::Mat eyeImg = Faceimg(cv::Rect(point1,point2));
	    EyeVec.push_back(eyeImg);  //将眼睛图像，放入容器中，用于后面的二分类
        }
    }
}


