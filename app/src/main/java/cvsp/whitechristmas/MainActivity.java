package cvsp.whitechristmas;

import java.util.List;



import org.opencv.android.BaseLoaderCallback;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;

import org.opencv.android.LoaderCallbackInterface;

import org.opencv.android.OpenCVLoader;

import org.opencv.core.Core;

import org.opencv.core.CvType;

import org.opencv.core.Mat;

import org.opencv.core.MatOfPoint;

import org.opencv.core.Rect;

import org.opencv.core.Scalar;

import org.opencv.core.Size;

import org.opencv.android.CameraBridgeViewBase;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import org.opencv.imgproc.Imgproc;



import android.app.Activity;

import android.os.Bundle;

import android.util.Log;

import android.view.MotionEvent;

import android.view.View;

import android.view.Window;

import android.view.WindowManager;

import android.view.View.OnTouchListener;

import android.view.SurfaceView;




public class MainActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {

    private static final String  TAG              = "MainActivity";

    private Mat                  mRgba;

//    static{ System.loadLibrary("opencv_java3"); }

    private CameraBridgeViewBase mOpenCvCameraView;

    private OpencvCalls opencvCalls;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {

        @Override

        public void onManagerConnected(int status) {

            switch (status) {

                case LoaderCallbackInterface.SUCCESS:

                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    mOpenCvCameraView.enableView();

                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);

                } break;

                default:

                {

                    super.onManagerConnected(status);

                } break;

            }

        }

    };



    public MainActivity() {

        Log.i(TAG, "Instantiated new " + this.getClass());
        System.loadLibrary("native-lib");
    }



    /** Called when the activity is first created. */

    @Override

    public void onCreate(Bundle savedInstanceState) {

        Log.i(TAG, "called onCreate");

        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);



        setContentView(R.layout .activity_main);


        //TODO: update name
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.color_blob_detection_activity_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

    }



    @Override

    public void onPause()

    {

        super.onPause();

        if (mOpenCvCameraView != null)

            mOpenCvCameraView.disableView();

    }



    @Override

    public void onResume()

    {

        super.onResume();

        if (!OpenCVLoader.initDebug()) {

            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");

            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);

        } else {

            Log.d(TAG, "OpenCV library found inside package. Using it!");

            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        }

    }



    public void onDestroy() {

        super.onDestroy();

        if (mOpenCvCameraView != null)

            mOpenCvCameraView.disableView();

    }



    public void onCameraViewStarted(int width, int height) {

        mRgba = new Mat(height, width, CvType.CV_8UC4);

    }



    public void onCameraViewStopped() {

        mRgba.release();

    }



    public boolean onTouch(View v, MotionEvent event) {


        return false; // don't need subsequent touch events

    }



    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        opencvCalls.faceDetection(mRgba.getNativeObjAddr());
        return mRgba;

    }



    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {

        Mat pointMatRgba = new Mat();

        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);

        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);



        return new Scalar(pointMatRgba.get(0, 0));

    }

}