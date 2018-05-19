#-------------------------------------------------
#
# Project created by QtCreator 2018-05-13T21:41:06
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FaceEye
TEMPLATE = app

LIBS += /usr/lib/x86_64-linux-gnu/libboost_date_time.a  \
        /usr/lib/x86_64-linux-gnu/libboost_system.a \
        -L/usr/local/lib -L/home/gjw/caffe-ssd/build/lib -L/usr/lib/x86_64-linux-gnu \
        -lopencv_highgui  -lopencv_core  -lopencv_imgproc  -lboost_system -lglog -lcaffe \
        -lprotobuf -lgflags -lswscale  -lboost_thread   -lpthread
INCLUDEPATH += \
            /usr/local/include \
            /usr/include/opencv \
            /usr/include/opencv2 \
            /home/gjw/caffe-ssd/include \
            /home/gjw/caffe-ssd/build/src \
            /usr/local/cuda-8.0/include    \
            /usr/include/boost/  \
            /usr/lib/x86_64-linux-gnu
SOURCES += main.cpp\
        mainwindow.cpp \
    ssd_detect.cpp

HEADERS  += mainwindow.h \
    classifer.hpp \
    ssd_detect.hpp

FORMS    += mainwindow.ui
