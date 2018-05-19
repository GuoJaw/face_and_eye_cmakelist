

##  1_制作眼睛数据集
        ## Intro
            C++，人脸检测，人眼检测--->制作人眼数据集：将眼睛小块保存到本地
        ## 使用技术
            人脸检测：SSD
            人眼检测：SSD
        ## Dependencies
            Caffe-SSD   

##  2_眼睛状态判断
	(1)EeyStatus_C++
	(2)EeyStatus_QT 
        ## Intro
            实现实时地睁眼闭眼判断，FPS≈37，在GTX1060下
        ## 使用技术
            人脸检测：MobileNet-SSD
            人眼检测：MobileNet-SSD
            眼睛睁开闭合状态（二分类）：Caffe-AlexNet
        ## Dependencies
            Caffe-MobileNet-SSD    

##  3_嵌套CMakelist写法
	1_生成.so库——功能介绍：
		[0] CMakeLists.txt：在bin下生成mobilenet_ssd可执行文件，最后add_subdirectory调用下面的三个子目录中的CMakelists.txt
		[1] classifer中的CMakelists.txt: 只用classifer.cpp生成动态库Classifer_SSD.so
		[2] ssd_detect中的CMakelists.txt: 只用ssd_detect.cpp生成动态库SSD_DETECT.so
		[3] eye_status中的CMakelists.txt： 用classifer.cpp，eyeStaus.cpp和ssd_detect.cpp生成动态库Final.so
	

	2_只使用Final.so库——功能介绍：用main.cpp调用Final.so中的test_eyeStates();函数
		[1] 只需要包含classifer，eyeStaus和ssd_detect头文件
		[2] 还需要包含Final.so（不需要包含.cpp实现，实现被封装到了Final.so动态库中）

