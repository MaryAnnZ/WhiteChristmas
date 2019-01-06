#include <jni.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <android/log.h>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core.hpp"
//#include "opencv2/objdetect/objdetect.hpp"


extern "C" JNIEXPORT jstring JNICALL
Java_cvsp_whitechristmas_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

//for converting from android frame to usable one
cv::Point convert(int x, int y, cv::Mat frame);
void doDeidentification(cv::Mat frame);
bool isInEllipse(int x, int y, cv::Rect rect, cv::Point center) ;
std::string type2str(int type);

std::vector<cv::Rect> faces;
cv::String face_cascade_name = "/storage/emulated/0/Download/haarcascade_frontalface_default.xml";
bool faceCascadedLoaded = false;
cv::CascadeClassifier face_cascade;
cv::Mat prevFrame, prevprevFrame;
int frameCounter = 0;
std::vector<std::vector<cv::Point2f>> keypoints;

extern "C"
JNIEXPORT void JNICALL
Java_cvsp_whitechristmas_OpencvCalls_faceDetection(JNIEnv *env, jclass type, jlong addrRgba) {

    cv::Mat &frame = *(cv::Mat *) addrRgba;     //8UC4
    cv::Mat frame_gray;

    auto started = std::chrono::high_resolution_clock::now();
    cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    cv::rotate(frame_gray, frame_gray, cv::ROTATE_90_CLOCKWISE);
    auto done = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
    __android_log_print(ANDROID_LOG_INFO, "Timer", "image processing");
    __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", time);

    if (frameCounter > 1) { //do tracking
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Track");

        std::vector<cv::Point2f> newKeypoints;
        std::vector<uchar> status;
        std::vector<float> error;
        std::vector<cv::Rect> facesTmp;
        for (unsigned int i = 0; i < keypoints.size(); ++i) {
            if (keypoints[i].size() > 0) {
                started = std::chrono::high_resolution_clock::now();
                cv::calcOpticalFlowPyrLK(prevprevFrame, prevFrame, keypoints[i], newKeypoints,
                                         status,
                                         error);
                done = std::chrono::high_resolution_clock::now();
                time = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
                __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint tracking");
                __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", time);
                keypoints[i].clear();
                float offsetX = (float) 0.0;
                float offsetY = (float) 0.0;
                int goodPixels = 0;
                for (unsigned int j = 0; j < newKeypoints.size(); ++j) {
                    uchar c = status[j];
                    if (c == 1) {

                        goodPixels++;
                        offsetX +=  (newKeypoints[j].x - keypoints[i][j].x);
                        offsetY +=  (newKeypoints[j].y - keypoints[i][j].y);
                        keypoints[i].push_back(newKeypoints[j]);
                        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "rendering points");
                        ellipse(frame, cv::Point(newKeypoints[j].y, frame.rows - newKeypoints[j].x),
                                cv::Size(5, 5), 0, 0, 360,
                                cv::Scalar(255, 0, 0), 4, 8, 0);
                    } else if (c == 0) {
                        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "status is 0");
                    }
                }
                if (goodPixels > 0) {
                    offsetX = offsetX / (float) goodPixels;
                    offsetY = offsetY / (float) goodPixels;
                    cv::Rect currentFace = faces[i];
                    faces[i] = cv::Rect(currentFace.x + (int)offsetX, currentFace.y + (int)offsetY, currentFace.width, currentFace.height);
                    cv::Point center(faces[i].y + faces[i].height * 0.5,
                                     frame.rows - faces[i].x - faces[i].width * 0.5);
                    ellipse(frame, center, cv::Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360,
                            cv::Scalar(255, 0, 255), 4, 8, 0);
                }
            }
        }
        newKeypoints.clear();
        prevprevFrame = prevFrame;
        prevFrame = frame_gray;

        if (frameCounter > 11) {
            frameCounter = 0;
            return;
        }
    } else { //do detection
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Detect");


        if (!faceCascadedLoaded) {
            if (!face_cascade.load(face_cascade_name)) {
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error loading face cascade");
                return;
            } else {
                faceCascadedLoaded = true;
            }
        }

        if (frameCounter == 0) {
            prevprevFrame = frame_gray;
            keypoints.clear();
        } else if (frameCounter == 1) {
            prevFrame = frame_gray;
        }
        //-- Detect faces
        faces.clear();
        started = std::chrono::high_resolution_clock::now();
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
                                      cv::Size(30, 30));
        done = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
        __android_log_print(ANDROID_LOG_INFO, "Timer", "face detection");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", time);
        std::vector<cv::Point2f> tmpKeypoints;

        //for ( int x = 0; x < frame_gray.rows; x++ )
        //{
        //    for ( int y = 0; y < frame_gray.cols; y++ )
        //    {
        //        //transform because of weird android bug
        //        cv::Point transPoint = convert(x,y,frame);
        //        if (transPoint.x>=0 && transPoint.x<frame.rows && transPoint.y>=0 && transPoint.y<frame.cols) {
        //            frame.at<cv::Vec3b>(transPoint) = frame_gray.at<cv::Vec3b>(cv::Point(x,y));
        //        }
        //    }
        //}

        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Faces Found: ");
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "%d", faces.size());
        for (size_t i = 0; i < faces.size(); i++) {

            cv::Rect currRect = faces[i];
            //RENDER DETECTED FACES
            cv::Point center(currRect.y + currRect.height * 0.5,
                             frame.rows - currRect.x - currRect.width * 0.5);
            ellipse(frame, center, cv::Size(currRect.width * 0.5, currRect.height * 0.5), 0, 0, 360,
                    cv::Scalar(255, 0, 255), 4, 8, 0);

            if (frameCounter == 0) {

                __android_log_print(ANDROID_LOG_INFO, "Keypoints", "Finding keypoints");
                cv::Mat mask = cv::Mat(frame_gray.rows, frame_gray.cols, CV_8UC1, (uchar) 0);
                mask(currRect) = 1;

                started = std::chrono::high_resolution_clock::now();
                cv::goodFeaturesToTrack(frame_gray, tmpKeypoints, 100, 0.2, 0.2, mask);
                done = std::chrono::high_resolution_clock::now();
                time = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
                __android_log_print(ANDROID_LOG_INFO, "Timer", "Feature Point Detection");
                __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", time);
                //render goodFeaturePoints
                for (unsigned int k = 0; k < tmpKeypoints.size(); k++) {
                    ellipse(frame, cv::Point(tmpKeypoints[k].y, frame.rows - tmpKeypoints[k].x),
                            cv::Size(5, 5), 0, 0, 360,
                            cv::Scalar(255, 0, 0), 4, 8, 0);
                }

                keypoints.push_back(tmpKeypoints);
            }

            if (keypoints.size() > 0) {
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Some keypoints found");
            } else {
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "No keypoints found");
            };

            //render keypoints
//            for (unsigned int i = 0; i < keypoints.size(); i++) {
//                for (unsigned int k = 0; k < keypoints[i].size(); k++) {
//                    ellipse(frame, cv::Point(keypoints[i][k].y, frame.rows - keypoints[i][k].x),
//                            cv::Size(5, 5), 0, 0, 360,
//                            cv::Scalar(255, 0, 0), 4, 8, 0);
//                }
//            }
        }
    }
//    if (faces.size() == 0) {
//        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "There are no faces");
//    }
//    for (size_t i = 0; i < faces.size(); i++) {
//        cv::Rect currRect = faces[i];
//        cv::Point center(currRect.y + currRect.height * 0.5,
//                         frame.rows - currRect.x - currRect.width * 0.5);
//        ellipse(frame, center, cv::Size(currRect.width * 0.5, currRect.height * 0.5), 0, 0, 360,
//                cv::Scalar(0, 255, 0), 4, 8, 0);
//    }

    doDeidentification(frame);

    frameCounter++;
}

cv::Point convert(int x, int y, cv::Mat frame) {
    cv::Point newPoint = cv::Point(0,0);
    newPoint.x  = y;
    newPoint.y  = frame.rows - x;
    return newPoint;
}

void doDeidentification(cv::Mat frame) {
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Rect currRect = faces[i];
        cv::Point center(currRect.x + currRect.width * 0.5, currRect.y + currRect.height * 0.5);
        //cv::Mat currentFace = image(currRect);

        //        medianBlur(src,dst,i);

        for(int y = currRect.y; y < currRect.y+currRect.height; y++)
        {
           for(int x = currRect.x; x < currRect.x+currRect.width; x++)
            {
               //if point is in the ellipse of rectancle, deindentificate!
               //if ((pow((x - center.x), 2) / pow(currRect.width, 2))  + (pow((y - center.y), 2) / pow(currRect.height, 2)) <1) {
                cv::Point transPoint = convert(x, y, frame);

               int grey = rand()%255 + 10;
               cv::Scalar intensity = frame.at<cv::Scalar>(transPoint);
               intensity.val[0] = (int)rand()%5+(grey);
               intensity.val[1]= (int)rand()%5+(grey);
               intensity.val[2]= (int)rand()%5+(grey);

                frame.at<cv::Vec4b>(transPoint) = cv::Vec4b(rand()%255, rand()%255, rand()%255,1);
            }
        }

        //for debugging
        //ellipse(frame, convert(center.x,center.y,frame), cv::Size(currRect.height * 0.5, currRect.width * 0.5), 0, 0, 360,cv::Scalar(rand()%255, rand()%255, rand()%255),CV_FILLED);
    }


}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
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
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error finding rect class");
        return nullptr;

    } else {
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "rect class created");

    }
    static jmethodID rectCtorID = env->GetMethodID(rectClass, "<init>", "(IIII)V");

//    if (rectCtorID == 0) {
//        rectClass = env->FindClass("org/opencv/core/Rect");
//        rectCtorID = env->GetMethodID(rectClass, "<init>", "(IIII)V");
//    }


    cv::Mat &frame = *(cv::Mat *) addrRgba;
    cv::String face_cascade_name = "/storage/emulated/0/Download/haarcascade_frontalface_default.xml";
    cv::CascadeClassifier face_cascade;

    if (!face_cascade.load(face_cascade_name)) {
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error loading face cascade");
        return nullptr;
    };

    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    cv::rotate(frame_gray, frame_gray, cv::ROTATE_90_CLOCKWISE);
    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
                                  cv::Size(30, 30));

    jobjectArray detectedFaces = env->NewObjectArray(faces.size(), (jclass) rectClass, 0);

    for (size_t i = 0; i < faces.size(); i++) {
        cv::Point center(faces[i].y + faces[i].height * 0.5,
                         frame.rows - faces[i].x - faces[i].width * 0.5);
        ellipse(frame, center, cv::Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360,
                cv::Scalar(255, 0, 255), 4, 8, 0);

        cv::Rect currentRect = faces[i];
        jobject rect = env->NewObject(rectClass, rectCtorID, currentRect.x, currentRect.y,
                                      currentRect.width, currentRect.height);
        env->SetObjectArrayElement(detectedFaces, i, rect);
    }
    return detectedFaces;
}