#Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.
set(CMAKE_C_COMPILER C:\\Users\\rebeb\\AppData\\Local\\Android\\sdk\\ndk-bundle\\toolchains\\x86-4.9\\prebuilt\\windows-x86_64\\bin\\i686-linux-android-gcc)
#Added 2 path definitions to support 20160825 additions
set(pathToProject D:\\DOKUMENTE\\UNI\\MA\\11.WS18\\CVUE\\WhiteChristmas)
set(pathToOpenCv D:\\DOKUMENTE\\UNI\\MA\\11.WS18\\CVUE\\WhiteChristmas\\OpenCV-android-sdk)

#Added by the IDE on project create
cmake_minimum_required(VERSION 3.4.1)

#Two sets suggested by Bruno Alexandre Krinski 20160825
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

#Addition suggested by Bruno Alexandre Krinski 20160825
include_directories(${pathToOpenCv}/sdk/native/jni/include)

#Added by IDE on project create
add_library( native-lib SHARED src/main/cpp/native-lib.cpp )

#Addition suggested by Bruno Alexandre Krinski 20160825
add_library( lib_opencv SHARED IMPORTED )

#Addition suggested by Bruno Alexandre Krinski 20160825
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${pathToProject}/app/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so)

#Added by IDE on project create
find_library( log-lib log )

#Added by IDE on project create, Removed and replace with additional parameter suggested by Bruno Alexandre Krinski 20160825
target_link_libraries( native-lib ${log-lib} lib_opencv)
