
aux_source_directory(. DIR_CLASSIFER_SRCS)  #编译当前目录中的所有CPP

### OpenCV的配置 ###
set (OpenCV_DIR /usr)  
find_package(OpenCV 2 REQUIRED)

# VC++目录（包含目录，库目录）
set (HEAD ${PROJECT_SOURCE_DIR}/include)
set (CAFFE_ROOT "/home/gjw/caffe-ssd")  # 设置常用变量：caffe根目录
  # 包含目录： 代码使用的头文件
set (CAFFE_INCLUDE ${CAFFE_ROOT}/include)
  # 库目录： 库.a或.so使用的头文件
set (CUDA_INCLUDE "/usr/local/cuda-8.0/include")
set (BOOST_SYSTEM_INCLUDE "/usr/include/boost")
set (GFLAGS_INCLUDE "/usr/local/include/gflags")
set (GLOG_INCLUDE "/usr/include/glog")

INCLUDE_DIRECTORIES(
		    ${HEAD}

		    ${CAFFE_INCLUDE}
		    ${CUDA_INCLUDE} ${BOOST_SYSTEM_INCLUDE} ${GFLAGS_INCLUDE} ${GLOG_INCLUDE} 
	   
		    ${OpenCV_INCLUDE_DIRS} 
)

# 生成SHARED链接库Classifer_SSD
add_library (Classifer_SSD SHARED ${DIR_CLASSIFER_SRCS})  
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) #设置生成的.so存放目录
 
# .so依赖的.so动态库
set (CAFFE_SSD_LIBS /home/gjw/caffe-ssd/build/lib/libcaffe.so)
set (BOOST_SYSTEM_LIBS "/usr/lib/x86_64-linux-gnu/libboost_system.so")
set (GFLAGS_LIBS "/usr/lib/x86_64-linux-gnu/libgflags.so")
set (GLOG_LIBS "/usr/local/lib/libglog.so")

# 链接Classifer_SSD依赖的库文件
target_link_libraries( 
	Classifer_SSD
	${CAFFE_SSD_LIBS} 
	${GFLAGS_LIBS} ${gtest_LIBS} ${GLOG_LIBS} ${BOOST_SYSTEM_LIBS}

	${OpenCV_LIBS} 
)


 








