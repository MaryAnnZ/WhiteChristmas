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
void convert(int x, int y, cv::Mat frame, cv::Point &oldPoint);
void doCallLogging();
void doDeidentification(cv::Mat currentFrame);

void recalculateDeidentification(cv::Mat frame_rot, cv::Mat frame);

void detectFace(cv::Mat frame_grey_rot, cv::Mat currentFrame);

void doTracking(cv::Mat frame_grey_rot, cv::Mat currentFrame);

void calcAndDrawKeyPointsForRealFace(cv::Mat frame_grey_rot, cv::Mat currentFrame);

cv::Rect realFace;
//cv::String face_cascade_name = "/storage/emulated/0/Download/haarcascade_frontalface_default.xml";
cv::String face_cascade_name = "/storage/emulated/0/Download/haarcascade_upperbody.xml";
//cv::String eyes_cascade_name = "/storage/emulated/0/Download/haarcascade_eye.xml";
cv::String eyes_cascade_name = "/storage/emulated/0/Download/haarcascade_frontalface_default.xml";
bool faceCascadedLoaded = false;
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;

cv::Mat prevFrame;
cv::Mat deIdentificationMask;   //mask overlay

std::vector<cv::Point2f> faceKeypoints;
int countKeypointsForFace;
float dumpedKeypoints = 1.0;
bool faceFound;

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
float grabCutTime = 0.0;
int grabCutCalls = 0;

std::vector<float> faceTrackingOffset; //offset since the beginning of tracking

bool faceTouched = false;
std::vector<int> faceLocation; //0 is x, 1 is y

int customFaceBBHeight = -1;
int customFaceMaxX = -1;
int customFaceMaxY = -1;

extern "C"
JNIEXPORT void JNICALL
Java_cvsp_whitechristmas_OpencvCalls_faceDetection(JNIEnv *env, jclass type, jlong addrRgba, jboolean backCamera) {
    cv::Mat &frame = *(cv::Mat *) addrRgba;     //8UC4

    auto started = std::chrono::high_resolution_clock::now();

    cv::Mat frame_rot = frame;
    cv::Mat frame_gray_rot;

    if (backCamera) {
        cv::rotate(frame_rot, frame_rot, cv::ROTATE_90_CLOCKWISE);
    } else {
        cv::rotate(frame_rot, frame_rot, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    cvtColor(frame_rot, frame_gray_rot, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray_rot, frame_gray_rot);

    if (customFaceBBHeight == -1) {
        customFaceBBHeight = (int) ((float) frame_gray_rot.rows * 0.15);
        customFaceMaxX = frame_gray_rot.cols - customFaceBBHeight;
        customFaceMaxY = frame_gray_rot.rows - customFaceBBHeight;
    }

    auto done = std::chrono::high_resolution_clock::now();
    preprocessTime += std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
    preprocessCalls++;

    if (dumpedKeypoints < 0.15 && !faceTouched && faceFound) {
        //if enough keypoints where found, and no user input in current frame, do tracking

        doTracking(frame_gray_rot, frame);

        __android_log_print(ANDROID_LOG_INFO, "Timer", "*****************");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint tracking");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "%f",
                            (keypointTrackingTime / (float) keypointTrackingCalls));
        __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint evaluation");
        __android_log_print(ANDROID_LOG_INFO, "Timer", "%f",
                            (evaluateKeypointsTime / (float) evaluateKeypointsCalls));
        __android_log_print(ANDROID_LOG_INFO, "Timer", "*****************");

    } else { //do detection

        faceTrackingOffset.clear();
        detectFace(frame_gray_rot, frame);
        recalculateDeidentification(frame_rot, frame);

        doCallLogging();

        faceTrackingOffset.push_back(0);
        faceTrackingOffset.push_back(0);
    }

    doDeidentification(frame);
}

void detectFace(cv::Mat frame_grey_rot, cv::Mat frame) {
    __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Detect");
    countKeypointsForFace = 0;
    faceKeypoints.clear();
    faceFound = false;

    if (!faceCascadedLoaded) {
        if (!face_cascade.load(face_cascade_name) ) {
            __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls",
                                "Error loading face cascade");
            return;
        } else if (!eyes_cascade.load(eyes_cascade_name)) {
            __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls",
                                "Error loading eye cascade");
        } else {
            faceCascadedLoaded = true;
        }
    }

    prevFrame = frame_grey_rot.clone();

    //-- Detect faces
    std::vector<cv::Rect> allFaces;
    auto started = std::chrono::high_resolution_clock::now();
    face_cascade.detectMultiScale(frame_grey_rot, allFaces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
                                  cv::Size(30, 30));
    auto done = std::chrono::high_resolution_clock::now();
    faceDetectionTime += std::chrono::duration_cast<std::chrono::milliseconds>(
            done - started).count();
    faceDetectionCalls++;

    if (faceTouched) {
        //get the only face which is at the point
        for (unsigned int i = 0; i < allFaces.size(); ++i) {
            cv::Rect currentFace = cv::Rect(allFaces[i]);
            if (currentFace.x < faceLocation[0] &&
                faceLocation[0] < (currentFace.x + currentFace.width) &&
                currentFace.y < faceLocation[1] &&
                faceLocation[1] < (currentFace.y + currentFace.height)) {

                realFace = currentFace;
                faceFound = true;

                break;
            }
        }

        faceLocation.clear();
        faceTouched = false;
        //no face were found at the given location
        /*       if (!faceFound) {
                   faces.clear();
                   int x = std::min(customFaceMaxX,
                                    std::max(0, (faceLocationX - (customFaceBBHeight / 2))));
                   int y = std::min(customFaceMaxY,
                                    std::max(0, (faceLocationY - (customFaceBBHeight / 2))));
                   cv::Rect face = cv::Rect(x, y, customFaceBBHeight, customFaceBBHeight);
                   faces.push_back(face);
                   faceLocationX = -1;
                   faceLocationY = -1;
               }*/
    }

    __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Faces Found: ");
    __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "%d", allFaces.size());
    started = std::chrono::high_resolution_clock::now();

    if (faceFound) {

        //draw touched face
        cv::Point center(realFace.y + realFace.height * 0.5,
                         frame.rows - realFace.x - realFace.width * 0.5);
        ellipse(frame, center, cv::Size(realFace.width * 0.5, realFace.height * 0.5), 0, 0, 360,
                cv::Scalar(255, 0, 255), 4, 8, 0);

        //get keypoints for face
        __android_log_print(ANDROID_LOG_INFO, "Keypoints", "Finding keypoints");
        calcAndDrawKeyPointsForRealFace(frame_grey_rot, frame);

    } else {
        bool eyesfound = false;
        for (size_t i = 0; i < allFaces.size() && !eyesfound; i++) {

            cv::Rect currRect = allFaces[i];
            //RENDER DETECTED FACES
            cv::Point center(currRect.y + currRect.height * 0.5,
                             frame.rows - currRect.x - currRect.width * 0.5);
            ellipse(frame, center, cv::Size(currRect.width * 0.5, currRect.height * 0.5), 0, 0, 360,
                    cv::Scalar(255, 0, 255), 4, 8, 0);

            //Check if face has eyes
            std::vector<cv::Rect> currenctFaceEyes;
            cv::Mat faceROI = frame_grey_rot(currRect);
            eyes_cascade.detectMultiScale(faceROI, currenctFaceEyes, 1.1, 2,
                                          0 | CV_HAAR_SCALE_IMAGE,
                                          cv::Size(30, 30));

            if (currenctFaceEyes.size() >= 1) { //at least two eyes are found
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Eyes In Face Found. ");
                realFace = currRect;
                eyesfound = true;
                faceFound = true;

                calcAndDrawKeyPointsForRealFace(frame_grey_rot, frame);
            }
        }

        done = std::chrono::high_resolution_clock::now();
        keypointFindingTime += std::chrono::duration_cast<std::chrono::milliseconds>(
                done - started).count();
        keypointFindingCalls++;

    }
}

void doTracking(cv::Mat frame_grey_rot, cv::Mat currentFrame) {
    __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Track");

    std::vector<cv::Point2f> newKeypoints;
    std::vector<uchar> status;
    std::vector<float> error;
    std::vector<cv::Rect> facesTmp;
    std::vector<cv::Point2f> keypointsToSave;

    if (faceKeypoints.size() > 0) {
        auto started = std::chrono::high_resolution_clock::now();
        cv::calcOpticalFlowPyrLK(prevFrame, frame_grey_rot, faceKeypoints, newKeypoints,
                                 status,
                                 error);
        auto done = std::chrono::high_resolution_clock::now();
        keypointTrackingTime += std::chrono::duration_cast<std::chrono::milliseconds>(
                done - started).count();
        keypointTrackingCalls++;

        keypointsToSave = std::vector<cv::Point2f>();
        float faceTrackingOffsetX = (float) 0.0;
        float faceTrackingOffsetY = (float) 0.0;
        int goodPixels = 0;
        started = std::chrono::high_resolution_clock::now();
        for (unsigned int j = 0; j < newKeypoints.size(); ++j) {
            uchar c = status[j];
            if (c == 1) {

                goodPixels++;
                faceTrackingOffsetX += (newKeypoints[j].x - faceKeypoints[j].x);
                faceTrackingOffsetY += (newKeypoints[j].y - faceKeypoints[j].y);
                keypointsToSave.push_back(newKeypoints[j]);
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "rendering points");
                ellipse(currentFrame,
                        cv::Point(newKeypoints[j].y, currentFrame.rows - newKeypoints[j].x),
                        cv::Size(5, 5), 0, 0, 360,
                        cv::Scalar(1 * 10, 0, 0), 4, 8, 0);
            } else if (c == 0) {
                __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "status is 0");
            }
        }
        if (goodPixels > 0) {
            float dumped = ((float) countKeypointsForFace - (float) goodPixels) /
                           (float) countKeypointsForFace;
            if (dumped > dumpedKeypoints) {
                dumpedKeypoints = dumped;
            }
            faceTrackingOffsetX = faceTrackingOffsetX / (float) goodPixels;
            faceTrackingOffsetY = faceTrackingOffsetY / (float) goodPixels;
            realFace = cv::Rect(realFace.x + (int) faceTrackingOffsetX,
                                realFace.y + (int) faceTrackingOffsetY, realFace.width,
                                realFace.height);

            cv::Point center(realFace.y + realFace.height * 0.5,
                             currentFrame.rows - realFace.x - realFace.width * 0.5);
            ellipse(currentFrame, center, cv::Size(realFace.width * 0.5, realFace.height * 0.5), 0,
                    0, 360,
                    cv::Scalar(255, 0, 255), 4, 8, 0);

            faceTrackingOffset[0] += faceTrackingOffsetX;
            faceTrackingOffset[1] += faceTrackingOffsetY;
        }
        done = std::chrono::high_resolution_clock::now();
        evaluateKeypointsTime += std::chrono::duration_cast<std::chrono::milliseconds>(
                done - started).count();
        evaluateKeypointsCalls++;
    }

    __android_log_print(ANDROID_LOG_INFO, "Keypoints", "Dumped keypoints");
    __android_log_print(ANDROID_LOG_INFO, "Keypoints", "%f", dumpedKeypoints);

    newKeypoints.clear();
    faceKeypoints = keypointsToSave;
    prevFrame = frame_grey_rot.clone();
}

void calcAndDrawKeyPointsForRealFace(cv::Mat frame_grey_rot, cv::Mat currentFrame) {
    std::vector<cv::Point2f> tmpKeypoints;

    cv::Mat mask = cv::Mat(frame_grey_rot.rows, frame_grey_rot.cols, CV_8UC1,
                           (uchar) 0);
    mask(realFace) = 1;

    cv::goodFeaturesToTrack(frame_grey_rot, tmpKeypoints, 100, 0.2, 0.2, mask);

    //render goodFeaturePoints
    for (unsigned int k = 0; k < tmpKeypoints.size(); k++) {
        ellipse(currentFrame, cv::Point(tmpKeypoints[k].y, currentFrame.rows - tmpKeypoints[k].x),
                cv::Size(5, 5), 0, 0, 360,
                cv::Scalar(255, 0, 0), 4, 8, 0);
    }

    faceKeypoints = tmpKeypoints;


    if (faceKeypoints.size() > 0) {
        countKeypointsForFace = tmpKeypoints.size() + 1;
        dumpedKeypoints = 0.0;
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Some keypoints found");
    } else {
        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "No keypoints found");
    };

    tmpKeypoints.clear();
}

void convert(int x, int y, cv::Mat frame, cv::Point &oldPoint) {
    oldPoint.x = y;
    oldPoint.y = frame.rows - x;
}

void doCallLogging() {
    __android_log_print(ANDROID_LOG_INFO, "Timer", "*****************");
    __android_log_print(ANDROID_LOG_INFO, "Timer", "preprocessing");
    __android_log_print(ANDROID_LOG_INFO, "Timer", "%f",
                        (preprocessTime / (float) preprocessCalls));
    __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint tracking");
    __android_log_print(ANDROID_LOG_INFO, "Timer", "%f",
                        (keypointTrackingTime / (float) keypointTrackingCalls));
    __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint evaluation");
    __android_log_print(ANDROID_LOG_INFO, "Timer", "%f",
                        (evaluateKeypointsTime / (float) evaluateKeypointsCalls));
    __android_log_print(ANDROID_LOG_INFO, "Timer", "face detection");
    __android_log_print(ANDROID_LOG_INFO, "Timer", "%f",
                        (faceDetectionTime / (float) faceDetectionCalls));
    __android_log_print(ANDROID_LOG_INFO, "Timer", "keypoint finding");
    __android_log_print(ANDROID_LOG_INFO, "Timer", "%f",
                        (keypointFindingTime / (float) keypointFindingCalls));
    __android_log_print(ANDROID_LOG_INFO, "Timer", "Grab Cut Segmentation");
    __android_log_print(ANDROID_LOG_INFO, "Timer", "%f",
                        (grabCutTime / (float) grabCutCalls));
    __android_log_print(ANDROID_LOG_INFO, "Timer", "*****************");
}



void doDeidentification(cv::Mat currentFrame) {
    if (faceFound) {
        cv::Point transPoint;
        for (int y = realFace.y-20; y < realFace.y + realFace.height+20; y++) {
            for (int x = realFace.x-20; x < realFace.x + realFace.width+20; x++) {
                unsigned char current = deIdentificationMask.at<unsigned char>(cv::Point(x-faceTrackingOffset[0], y-faceTrackingOffset[1]));

                //http://answers.opencv.org/question/3031/smoothing-with-a-mask/
                if (current == 1 || current == 3) {
                    convert(x, y, currentFrame, transPoint);
                    currentFrame.at<cv::Vec4b>(transPoint) = cv::Vec4b(rand() % 255, rand() % 255,  rand() % 255, 1);
                }
            }

        }

        cv::Point center(realFace.y + realFace.height * 0.5,
                         currentFrame.rows - realFace.x - realFace.width * 0.5);
        ellipse(currentFrame, center, cv::Size(realFace.width * 0.5, realFace.height * 0.5), 0,
                0, 360,
                cv::Scalar(255, 0, 0), 4, 8, 0);
    }
}

void recalculateDeidentification(cv::Mat frame_rot, cv::Mat frame) {
    if (faceFound) {

        cv::Rect bounds = realFace;
        bounds+= cv::Size(20, 20);
        cv::Mat matRect = frame_rot;
        cv::cvtColor(matRect, matRect, CV_BGRA2BGR);
        cv::Mat result = cv::Mat::ones(matRect.rows, matRect.cols, CV_8U*2); // all 3
        result(bounds) = 3;
        cv::Mat bgModel, fgModel; // the models (internally used)

        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Calculating Faceboundaries");
        auto started = std::chrono::high_resolution_clock::now();


        // GrabCut segmentation
        cv::grabCut(matRect,    // input image = the rect with face
                    result,
                    realFace,// rectangle containing foreground
                    bgModel, fgModel, // models
                    5,        // number of iterations
                    cv::GC_INIT_WITH_RECT ); // use rectangle
        auto done = std::chrono::high_resolution_clock::now();
        grabCutTime += std::chrono::duration_cast<std::chrono::milliseconds>(
                done - started).count();
        grabCutCalls++;

        __android_log_print(ANDROID_LOG_INFO, "OpenCVCalls", "Done Calculating Faceboundaries");


        deIdentificationMask = result;

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

    faceTouched = true;
    faceLocation.push_back(std::max(0, x - 150));
    faceLocation.push_back(prevFrame.cols - y);
}