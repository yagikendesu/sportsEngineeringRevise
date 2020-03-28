
// * @file main.cpp
// * @brief 100m走の動画から各選手の歩幅、歩数を算出
// * @author 八木賢太郎
// * @date 2018/1/3
// */

#include <opencv2/opencv.hpp>
#include "src/panorama.h"
#include "src/openpose/myOpenPose.h"
#include "src/basicFunctions/basicFunction.h"
#include <ceres-solver/include/ceres/ceres.h>

//#include <openpose/flags.hpp>
//#include "src/nDegreeApproximation.h"
//#include "src/basicFunctions/basicFunction.h"
using namespace std;
using namespace yagi;
using namespace cv;


int main() {
//    string folder_path = "/home/yagi/sfmDR/inputVideos/" + video_name + "/";
 Panorama Panorama("woman_2015_yosen_5");
 Panorama.INIT_PROCESSING = false;
 Panorama.videoType = ".mp4";
    Panorama.USE_LASTMASK = false;
    Panorama.USE_LAST_TRACKLINE = false;
    Panorama.USE_LAST_CORNERS = false;
     Panorama.SELECT_TARGET_RANE = false;
     Panorama.MAX_TRANSLATION = 3;
     Panorama.FPS = 25;
     Panorama.START_FRAME_DIST = 2;
     Panorama.TARGET_RUNNER_ID = 6;
     Panorama.TRACKING_MIN_DIST = 40;
     Panorama.STEP_JUDGE_RANGE = 5;
     Panorama.MASK_MARGIN = 50;
     Panorama.OP_MASK_RADIUS = 15;
     Panorama.FIRST_IM_ID = 10;
     Panorama.LAST_IM_ID = 245;
     Panorama.PROJECTION_STEP = 10;
     Panorama.RANSAC_LOOP_LIMIT = 10;
     Panorama.RANSAC_INLIER_RANGE = 1;
     Panorama.STROBO_RESIZE_MARGIN = 30;
     Panorama.SHOW_LOADED_IMAGE = false;
     Panorama.SHOW_ONLINE_POINTS = false;
     Panorama.SHOW_MASK_REGIONS = false;
     Panorama.SHOW_TRANSLATION = false;
     Panorama.SHOW_HOMOGRAPHY = false;
     Panorama.SHOW_TRACKLINES = false;
     Panorama.SHOW_PANORAMA = false;
     Panorama.SHOW_TRACKING_RUNNER = false;
     Panorama.SHOW_STROBO_PROCESS = false;
     Panorama.SHOW_RUNNER_CANDIDATES = false;
 //    Panorama.ESTIMATE_STEPS = false;
     Panorama.GENERATE_STROBO = false;
     Panorama.GENERATE_VIRTUALRACE = false;
     Panorama.VIRTUAL_TARGET_VIDEO = "Bolt958";
 //    Panorama.REMOVE_OTHER_RUNNERS = false;

 //    Panorama.setVariables(video_name);
     if (!checkFileExistence(Panorama._image_list_path)) {
         Panorama.videotoImage();
         outputTextFromVideo(Panorama._video_folder + Panorama._video_name + Panorama.videoType,
                             Panorama._project_path + "/openpose_image/", Panorama._txt_folder);
     }
     Panorama.loadingData();
     Panorama.masking();
     Panorama.trackDetection();
     Panorama.makePanorama();
     Panorama.trackingRunner();

 //    Panorama.saveData();
     if(Panorama.GENERATE_STROBO)
         Panorama.makeStroboRangeImage();
     if(Panorama.ESTIMATE_STEPS)
         Panorama.estimateStepPoints();
     if(Panorama.GENERATE_VIRTUALRACE)
         Panorama.makeVirtualRaceImages();
     return 0;
}
