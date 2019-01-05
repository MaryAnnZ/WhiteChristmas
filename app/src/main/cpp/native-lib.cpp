#include <jni.h>
#include <string>
#include <iostream>
#include <fstream>
#include <android/log.h>
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
//#include "opencv2/objdetect/objdetect.hpp"


extern "C" JNIEXPORT jstring JNICALL
Java_cvsp_whitechristmas_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}


std::vector<cv::Rect> faces;
cv::String face_cascade_name = "/storage/emulated/0/Download/haarcascade_frontalface_default.xml";
bool faceCascadedLoaded = false;
cv::CascadeClassifier face_cascade;
cv::Mat prevFrame, prevprevFrame;
int frameCounter = 0;
extern "C"
JNIEXPORT void JNICALL
Java_cvsp_whitechristmas_OpencvCalls_faceDetection(JNIEnv *env, jclass type, jlong addrRgba) {


        cv::Mat &frame = *(cv::Mat *) addrRgba;

        if (!faceCascadedLoaded) {
            if (!face_cascade.load(face_cascade_name)) {
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error loading face cascade");
                return;
            } else {
                faceCascadedLoaded = true;
            }
        }
        cv::Mat frame_gray;

        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        cv::equalizeHist(frame_gray, frame_gray);
        cv::rotate(frame_gray, frame_gray, cv::ROTATE_90_CLOCKWISE);

        if (frameCounter == 0) {
            prevprevFrame = frame;
        } else if (frameCounter == 1) {
            prevFrame = frame;
        }
        //-- Detect faces
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
                                      cv::Size(30, 30));

        for (size_t i = 0; i < faces.size(); i++) {
            cv::Point center(faces[i].y + faces[i].height * 0.5,
                             frame.rows - faces[i].x - faces[i].width * 0.5);
            ellipse(frame, center, cv::Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360,
                    cv::Scalar(255, 0, 255), 4, 8, 0);

        }

}

extern "C"
JNIEXPORT void JNICALL
Java_cvsp_whitechristmas_OpencvCalls_faceTracking(JNIEnv *env, jclass type) {

    // TODO

}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_cvsp_whitechristmas_OpencvCalls_getFaces__J(JNIEnv *env, jclass type, jlong addrRgba) {
    static jclass rectClass = env->FindClass("org/opencv/core/Rect");
    if (rectClass == NULL) {
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error finding rect class"); return nullptr;

    } else {
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "rect class created");

    }
    static jmethodID rectCtorID = env->GetMethodID(rectClass, "<init>", "(IIII)V");

//    if (rectCtorID == 0) {
//        rectClass = env->FindClass("org/opencv/core/Rect");
//        rectCtorID = env->GetMethodID(rectClass, "<init>", "(IIII)V");
//    }


    cv::Mat& frame = *(cv::Mat*)addrRgba;
    cv::String face_cascade_name = "/storage/emulated/0/Download/haarcascade_frontalface_default.xml";
    cv::CascadeClassifier face_cascade;

    if( !face_cascade.load( face_cascade_name ) ){ __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error loading face cascade"); return nullptr; };

    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );
    cv::rotate(frame_gray, frame_gray, cv::ROTATE_90_CLOCKWISE);
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    jobjectArray detectedFaces = env->NewObjectArray(faces.size(),(jclass)rectClass, 0);

    for( size_t i = 0; i < faces.size(); i++ )
    {
        cv::Point center( faces[i].y + faces[i].height*0.5, frame.rows-faces[i].x - faces[i].width*0.5 );
        ellipse( frame, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );

        cv::Rect currentRect = faces[i];
        jobject rect = env->NewObject(rectClass, rectCtorID, currentRect.x, currentRect.y, currentRect.width, currentRect.height);
        env->SetObjectArrayElement(detectedFaces, i, rect);
    }
    return detectedFaces;
}