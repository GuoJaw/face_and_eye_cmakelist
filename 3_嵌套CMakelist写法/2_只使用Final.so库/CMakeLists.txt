#1. 预备工作
cmake_minimum_required(VERSION 2.8) # cmake verson，指定cmake版本 
add_definitions(-std=c++11 -g)  # 非常关键：在CMake中使用C++11
project (mobilenet_ssd) # 项目名字

#2. 
   # 项目使用的所有的 .c  .cpp
set(source_files
	main.cpp
)
   # 项目使用的所有的 .h  .hpp
set(project_include
	include
	classifer
	ssd_detect

	eye_status   #####
)

### OpenCV的配置 ###
set (OpenCV_DIR /usr)  
find_package(OpenCV 2 REQUIRED)


#3. VC++目录（包含目录，库目录）
set (CAFFE_ROOT /home/gjw/caffe-ssd)  # 设置常用变量：caffe根目录
  #3-1. 包含目录： 代码使用的头文件
set (CAFFE_INCLUDE ${CAFFE_ROOT}/include)
  #3-2. 库目录： 库.a或.so使用的头文件
set (CUDA_INCLUDE /usr/local/cuda-8.0/include)
set (BOOST_SYSTEM_INCLUDE /usr/include/boost)
set (GFLAGS_INCLUDE /usr/local/include/gflags)
set (GLOG_INCLUDE /usr/include/glog)

INCLUDE_DIRECTORIES(
		    ${project_include} 
 
		    ${CAFFE_INCLUDE}
		    ${CUDA_INCLUDE} ${BOOST_SYSTEM_INCLUDE} ${GFLAGS_INCLUDE} ${GLOG_INCLUDE} 
	   
		    ${OpenCV_INCLUDE_DIRS} 
)



set (CAFFE_SSD_LIBS ${CAFFE_ROOT}/build/lib/libcaffe.so)
set (BOOST_SYSTEM_LIBS /usr/lib/x86_64-linux-gnu/libboost_system.so)
set (GFLAGS_LIBS /usr/lib/x86_64-linux-gnu/libgflags.so)
set (GLOG_LIBS /usr/local/lib/libglog.so)

#4. add_executable生成可执行文件
add_executable (mobilenet_ssd main.cpp  ${source_files} ) # 生成可执行文件
SET_TARGET_PROPERTIES( # 设定可执行文件的生成目录 
	mobilenet_ssd  PROPERTIES RUNTIME_OUTPUT_DIRECTORY  
	${PROJECT_SOURCE_DIR}/bin   # cmake..后，会自动生成bin目录
) 

set (FINAL_LIBS ${PROJECT_SOURCE_DIR}/lib/libFinal.so)###

target_link_libraries( ## 可执行程序mobilenet_ssd，在执行时，调用下面的dll库
	mobilenet_ssd 

	${FINAL_LIBS}
	
	${CAFFE_SSD_LIBS} 
	${GFLAGS_LIBS} ${gtest_LIBS} ${GLOG_LIBS} ${BOOST_SYSTEM_LIBS}

	${OpenCV_LIBS} 
)











