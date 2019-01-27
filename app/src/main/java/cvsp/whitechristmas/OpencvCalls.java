package cvsp.whitechristmas;

import org.opencv.core.Rect;

import java.util.ArrayList;

public class OpencvCalls {
    public native static void faceDetection(long addrRgba, boolean backCamera);
    public native static Rect[] getFaces(long addrRgba);
    public native static void setFaceLocation(int x, int y);
    public native static void faceTracking();
}
