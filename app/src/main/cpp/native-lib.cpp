#include <jni.h>
#include <string>
#include <iostream>
#include <fstream>
#include <android/log.h>
#include "opencv2/opencv.hpp"
//#include "opencv2/objdetect/objdetect.hpp"


extern "C" JNIEXPORT jstring JNICALL
Java_cvsp_whitechristmas_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}



extern "C"
JNIEXPORT void JNICALL
Java_cvsp_whitechristmas_OpencvCalls_faceDetection(JNIEnv *env, jclass type, jlong addrRgba) {

    cv::Mat& frame = *(cv::Mat*)addrRgba;
    cv::String face_cascade_name = "/storage/emulated/0/Download/haarcascade_frontalface_default.xml";
    cv::String eyes_cascade_name = "/storage/emulated/0/Download/haarcascade_eye.xml";
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    if( !face_cascade.load( face_cascade_name ) ){ __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error loading face cascade"); return; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error loading eye cascade"); return; };


    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );
    cv::rotate(frame_gray, frame_gray, cv::ROTATE_90_CLOCKWISE);
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
//        cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        cv::Point center( faces[i].y + faces[i].height*0.5, frame.rows-faces[i].x - faces[i].width*0.5 );
        ellipse( frame, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );

        cv::Mat faceROI = frame_gray( faces[i] );
        std::vector<cv::Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
        {
//            cv::Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
            cv::Point center( faces[i].y + eyes[j].y + eyes[j].height*0.5, frame.rows-eyes[i].x - eyes[i].width*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, center, radius, cv::Scalar( 255, 0, 0 ), 4, 8, 0 );

        }
    }

}