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
void convert(int x, int y, cv::Mat frame, cv::Point& oldPoint);
void doDeidentification(cv::Mat frame,cv::Mat rotated_gray);

std::vector<cv::Rect> faces;
cv::String face_cascade_name = "/storage/emulated/0/Download/haarcascade_frontalface_default.xml";
cv::String eyes_cascade_name = "/storage/emulated/0/Download/haarcascade_eye.xml";
bool faceCascadedLoaded = false;
cv::CascadeClassifier face_cascade;
cv::Mat prevFrame;
cv::CascadeClassifier eyes_cascade;

std::vector<std::vector<cv::Point2f>> keypoints;
std::vector<int> countKeypointsPerFace;
float dumpedKeypoints = 1.0;

float preprocessTime = 0.0;
int preprocessCalls = 0;
float keypointTrackingTime = 0.0;
int keypointTrackingCalls = 0;
float evaluateKeypointsTime = 0.0;
int evaluateKeypointsCalls = 0;
float faceDetectionTime = 0.0;
int faceDetectionCalls = 0;
float keypointFindingTime = 0.0;
int keypointFindingCalls = 0;

int faceLocationX = -1;
int faceLocationY = -1;
int customFaceBBHeight = -1;
int customFaceMaxX = -1;
int customFaceMaxY = -1;
extern "C"
JNIEXPORT void JNICALL
Java_cvsp_whitechristmas_OpencvCalls_faceDetection(JNIEnv *env, jclass type, jlong addrRgba) {
    cv::Mat &frame = *(cv::Mat *) addrRgba;     //8UC4

    auto started = std::chrono::high_resolution_clock::now();

    cv::Mat frame_rot = frame;
    cv::Mat frame_gray;
    cv::rotate(frame_rot, frame_rot, cv::ROTATE_90_CLOCKWISE);
    cvtColor(frame_rot, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    if (customFaceBBHeight == -1) {
        customFaceBBHeight = (int)((float) frame_gray.rows * 0.15);
        customFaceMaxX = frame_gray.cols - customFaceBBHeight;
        customFaceMaxY = frame_gray.rows - customFaceBBHeight;
    }


    auto done = std::chrono::high_resolution_clock::now();
    preprocessTime += std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
    preprocessCalls++;

    if (dumpedKeypoints < 0.15 && faceLocationX == -1) { //do tracking
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Track");

        std::vector<cv::Point2f> newKeypoints;
        std::vector<uchar> status;
        std::vector<float> error;
        std::vector<cv::Rect> facesTmp;
        std::vector<std::vector<cv::Point2f>> keypointsToSave;
        for (unsigned int i = 0; i < keypoints.size(); ++i) {
            if (keypoints[i].size() > 0) {
                started = std::chrono::high_resolution_clock::now();
                cv::calcOpticalFlowPyrLK(prevFrame, frame_gray, keypoints[i], newKeypoints,
                                         status,
                                         error);
                done = std::chrono::high_resolution_clock::now();
                keypointTrackingTime += std::chrono::duration_cast<std::chrono::milliseconds>(
                        done - started).count();
                keypointTrackingCalls++;
                keypointsToSave.push_back(std::vector<cv::Point2f>());
                float offsetX = (float) 0.0;
                float offsetY = (float) 0.0;
                int goodPixels = 0;
                started = std::chrono::high_resolution_clock::now();
                for (unsigned int j = 0; j < newKeypoints.size(); ++j) {
                    uchar c = status[j];
                    if (c == 1) {

                        goodPixels++;
                        offsetX += (newKeypoints[j].x - keypoints[i][j].x);
                        offsetY += (newKeypoints[j].y - keypoints[i][j].y);
                        keypointsToSave[i].push_back(newKeypoints[j]);
                        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "rendering points");
                        ellipse(frame, cv::Point(newKeypoints[j].y, frame.rows - newKeypoints[j].x),
                                cv::Size(5, 5), 0, 0, 360,
                                cv::Scalar(i * 10, 0, 0), 4, 8, 0);
                    } else if (c == 0) {
                        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "status is 0");
                    }
                }
                if (goodPixels > 0) {
                    float dumped = ((float) countKeypointsPerFace[i] - (float) goodPixels) / (float) countKeypointsPerFace[i];
                    if (dumped > dumpedKeypoints) {
                        dumpedKeypoints = dumped;
                    }
                    offsetX = offsetX / (float) goodPixels;
                    offsetY = offsetY / (float) goodPixels;
                    cv::Rect currentFace = faces[i];
                    faces[i] = cv::Rect(currentFace.x + (int) offsetX,
                                        currentFace.y + (int) offsetY, currentFace.width,
                                        currentFace.height);
                    cv::Point center(faces[i].y + faces[i].height * 0.5,
                                     frame.rows - faces[i].x - faces[i].width * 0.5);
                    ellipse(frame, center, cv::Size(faces[i].width * 0.5, faces[i].height * 0.5), 0,
                            0, 360,
                            cv::Scalar(255, 0, 255), 4, 8, 0);
                }
                done = std::chrono::high_resolution_clock::now();
                evaluateKeypointsTime += std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
                evaluateKeypointsCalls++;
            }
        }
        __android_log_print(ANDROID_LOG_INFO, "Keypoints", "Dumped keypoints");
        __android_log_print(ANDROID_LOG_INFO, "Keypoints", "%f", dumpedKeypoints);
        newKeypoints.clear();
        keypoints=keypointsToSave;
        prevFrame = frame_gray.clone();

    } else { //do detection
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Detect");
        countKeypointsPerFace.clear();
        keypoints.clear();
        if (!faceCascadedLoaded) {
            if (!face_cascade.load(face_cascade_name)) {
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Error loading face cascade");
                return;
            } else {
                faceCascadedLoaded = true;
            }
        }


        prevFrame = frame_gray.clone();

        //-- Detect faces
        faces.clear();
        started = std::chrono::high_resolution_clock::now();
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
                                      cv::Size(30, 30));
        done = std::chrono::high_resolution_clock::now();
        faceDetectionTime += std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
        faceDetectionCalls++;

        //get the only face which is at the point
        bool faceFound = false;
        if (faceLocationX != -1 && faceLocationY != -1) {
            for (unsigned int i = 0; i < faces.size(); ++i) {
                cv::Rect face = cv::Rect(faces[i]);
                if (face.x < faceLocationX && faceLocationX < (face.x + face.width) && face.y < faceLocationY && faceLocationY < (face.y + face.height)) {
                    faces.clear();
                    faces.push_back(face);
                    faceLocationX = -1;
                    faceLocationY = -1;
                    faceFound = true;
                    break;
                }
            }
            //no face were found at the given location
            if (!faceFound) {
                faces.clear();
                int x = std::min(customFaceMaxX,
                                 std::max(0, (faceLocationX - (customFaceBBHeight / 2))));
                int y = std::min(customFaceMaxY,
                                 std::max(0, (faceLocationY - (customFaceBBHeight / 2))));
                cv::Rect face = cv::Rect(x, y, customFaceBBHeight, customFaceBBHeight);
                faces.push_back(face);
                faceLocationX = -1;
                faceLocationY = -1;
            }
        }

        std::vector<cv::Point2f> tmpKeypoints;
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Faces Found: ");
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "%d", faces.size());
        started = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < faces.size(); i++) {

            cv::Rect currRect = faces[i];
            //RENDER DETECTED FACES
            cv::Point center(currRect.y + currRect.height * 0.5,
                             frame.rows - currRect.x - currRect.width * 0.5);
            ellipse(frame, center, cv::Size(currRect.width * 0.5, currRect.height * 0.5), 0, 0, 360,
                    cv::Scalar(255, 0, 255), 4, 8, 0);

			//-- In each face, detect eyes
            //std::vector<cv::Rect> eyes;
            //cv::Mat faceROI = frame_gray(currRect);
            //eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

            __android_log_print(ANDROID_LOG_INFO, "Keypoints", "Finding keypoints");
            cv::Mat mask = cv::Mat(frame_gray.rows, frame_gray.cols, CV_8UC1, (uchar) 0);
            mask(currRect) = 1;

            cv::goodFeaturesToTrack(frame_gray, tmpKeypoints, 100, 0.2, 0.2, mask);
            //render goodFeaturePoints
            for (unsigned int k = 0; k < tmpKeypoints.size(); k++) {
                ellipse(frame, cv::Point(tmpKeypoints[k].y, frame.rows - tmpKeypoints[k].x),
                        cv::Size(5, 5), 0, 0, 360,
                        cv::Scalar(255, 0, 0), 4, 8, 0);
            }
            keypoints.push_back(tmpKeypoints);


            if (keypoints.size() > 0) {
                countKeypointsPerFace.push_back(tmpKeypoints.size()+1);
                dumpedKeypoints = 0.0;
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Some keypoints found");
            } else {
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "No keypoints found");
            };
        }
        done = std::chrono::high_resolution_clock::now();
        keypointFindingTime += std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
        keypointFindingCalls++;

        __android_log_print(ANDROID_LOG_INFO, "Timer", "preprocessing");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", (preprocessTime/(float) preprocessCalls));
        __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint tracking");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", (keypointTrackingTime/(float) keypointTrackingCalls));
        __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint evaluation");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", (evaluateKeypointsTime/(float) evaluateKeypointsCalls));
        __android_log_print(ANDROID_LOG_INFO, "Timer", "face detection");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", (faceDetectionTime/(float) faceDetectionCalls));
        __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint finding");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "%f", (keypointFindingTime/(float) keypointFindingCalls));
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

//    doDeidentification(frame,frame_rot);
}

void convert(int x, int y, cv::Mat frame, cv::Point& oldPoint) {
    oldPoint.x = y;
    oldPoint.y =frame.rows - x;


}

void doDeidentification(cv::Mat frame,cv::Mat frame_rot) {
    cv::Point transPoint;

    if (faces.size()>0) {
        cv::Rect currRect = faces[1];
        cv::Mat matRect = frame_rot;
        cv::cvtColor(matRect,matRect,CV_BGRA2BGR);
        cv::Mat result = cv::Mat::zeros(matRect.rows, matRect.cols, CV_8U); // all 0
        cv::Mat bgModel, fgModel; // the models (internally used)

			__android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Doing Deidentification");

        // GrabCut segmentation
        cv::grabCut(matRect,    // input image = the rect with face
                    result,
                    currRect,// rectangle containing foreground
                    bgModel, fgModel, // models
                    1,        // number of iterations
                    cv::GC_INIT_WITH_RECT); // use rectangle

        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Doing 2");

        for(int y = currRect.y; y < currRect.y+currRect.height; y++)
        {
           for(int x = currRect.x; x < currRect.x+currRect.width; x++)
            {
                               convert(x, y, frame,transPoint);
                               frame.at<cv::Vec4b>(transPoint) = result.at<cv::Vec4b>(cv::Point(x, y));
            }
        }

    }


//    for (size_t i = 0; i < faces.size(); i++) {
//        //cv::Rect currRect = faces[i];
//        //cv::Point center(currRect.x + currRect.width * 0.5, currRect.y + currRect.height * 0.5);
//        //cv::Mat currentFace = image(currRect);
//        //for(int y = currRect.y; y < currRect.y+currRect.height; y++)
//        //{
//         //  for(int x = currRect.x; x < currRect.x+currRect.width; x++)
//        //    {
//        //                       convert(x, y, frame,transPoint);
//        //                        cv::grabCut()
//        //                       frame.at<cv::Vec4b>(transPoint) = cv::Vec4b(rand()%255, rand()%255, rand()%255,1);
//        //    }
//        //}
//
//        //for debugging
//        //ellipse(frame, convert(center.x,center.y,frame), cv::Size(currRect.height * 0.5, currRect.width * 0.5), 0, 0, 360,cv::Scalar(rand()%255, rand()%255, rand()%255),CV_FILLED);
//    }

}

  std::string r;
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

extern "C"
JNIEXPORT void JNICALL
Java_cvsp_whitechristmas_OpencvCalls_setFaceLocation(JNIEnv *env, jclass type, jint x, jint y) {
    if (prevFrame.empty()) {
        return;
    }
    faceLocationY = std::max(0, x - 150);
    faceLocationX = prevFrame.cols - y;
}