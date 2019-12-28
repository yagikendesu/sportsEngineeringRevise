#pragma once

#include <./opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include "videoToImage/videoToImage.h"
#include "videoToImage/trimVideo.h"


namespace yagi {

    class Panorama {

    public:
        std::string _video_name;
        std::string _project_path;
        std::string _image_folder;
        std::string _result_folder;
        std::string _video_folder;
        std::string _txt_folder;
        std::string _image_list_path;
        std::string _openpose_list_path;


        int MASK_MARGIN;
        bool SHOW_TRACKING_RUNNER;
        int MAX_TRANSLATION;
        int FPS;
        int START_FRAME_DIST;
        bool SHOW_TRANSLATION;
        bool SHOW_HOMOGRAPHY;
        bool SHOW_TRACKLINES;
        bool SHOW_STROBO_PROCESS;
        bool SHOW_PANORAMA;
        bool SHOW_RUNNER_CANDIDATES;
        bool SHOW_MASK_REGIONS;
        int OP_MASK_RADIUS;
        int FIRST_IM_ID;
        int LAST_IM_ID;
        int PROJECTION_STEP;
        bool ESTIMATE_STEPS = true;
        bool GENERATE_STROBO = true;
        bool GENERATE_VIRTUALRACE = true;
        int RANSAC_LOOP_LIMIT;
        int RANSAC_INLIER_RANGE;
        int STROBO_RESIZE_MARGIN;
        bool REMOVE_OTHER_RUNNERS;
        std::string VIRTUAL_TARGET_VIDEO;

        Panorama(std::string video_name);

        ~Panorama(){};

        //変数格納
        void setVariables(std::string video_name);

        void correctSteps();
        void trackMask(cv::Mat& im);
        void videotoImage();
        cv::Point2f templateMatching(cv::Mat& im1, cv::Mat& im2, cv::Rect tempRect, cv::Mat& maskImage, cv::Point2f preTranslation,  const int frameID);
        void getTranslationByTempMatching();
        void getTranslationByMyTempMatching();
        void getTranslationByBatchTempMatching();
        void getTranslationByOpticalFlow();
        void featurePointFindHomography();
        void maskingRunners();
        void trackingUsingHead();
        bool checkMin(int imID, int laneID);
        bool checkInLaneLines(int imID, int laneID);
        void loadingData();
        void loadImage();
        void masking();
        void calcMeanStep();


        cv::Mat  calcOpenPoseMask(std::vector<cv::Point2f>& pts, cv::Size imSize);

        void selectTrack();


        void trackTracking();


        void selectRunnerCandidates();


        void makePanorama();

        void selectMaskArea();


        void trackDetection();

        void trackingRunner();

        void detectHumanArea();


        void generatePanorama();

        void estimateStepPoints();
        void maskHumanArea();
        void getOpenPoseMask();


        void setMaskArea(cv::Rect _mask_area, cv::Point2i _center);


        void getOverviewHomography();


        void measuringStepPositions();


        void projectTrackLine();


        void trackTargetRunner();


        void startFinishLineSelect();



        void makeStroboImage();


        void generateNthFramePanorama();


        void makeVirtualRaceImages();


        void templateInverseMatchingFindHomography();


        void generateInversePanorama();

        void saveData();

        void getInverseOverviewHomography();

        void projectInverseTrackLine();

        void legLaneDist();

        void candidateStepFrame();

        void insideLane();

        void mergeStepID();

        void visualizeSteps();

        void getAllScaleFootPosition();

        void averageCompletion();

        void calculateStrideLength();

        void calcOpticalFlow();

        void visualizeStride();

        void getHomographyFromTranslation();

        void curveFitting(std::vector<double> time10mList);

        void getTranslation();

        void translateImage();

        void makeStroboRangeImage();

        void pitchCompletion();

        cv::Rect getMaskRect(std::vector<cv::Point2f>& body);

        //
        bool INIT_PROCESSING;
        bool SHOW_LOADED_IMAGE;
        bool SHOW_ONLINE_POINTS;
        bool USE_LASTMASK = true;
        bool USE_LAST_TRACKLINE = true;
        bool USE_LAST_CORNERS = true;
        bool SELECT_TARGET_RANE;
        int IMG_WIDTH;
        int IMG_HEIGHT;
        std::string videoType;
        //
        std::vector<std::string> img_names;
        cv::Mat resizeH;

        //OpenPoseBody
        class OpenPoseBody {
        public:
            //Methodｓ
            OpenPoseBody() {};
            ~OpenPoseBody() {};
            void setBodyCoord(std::vector<std::string> coord);
            std::vector<cv::Point2f> getBodyCoord();
            void clearBodyCoord();
            void setMaskRect();

            //Variables
            cv::Rect mask_rect;
            int humanID = 100;
            std::vector<cv::Point2f> bodyPts;
            std::vector<float> _confidenceMap;
            cv::Mat openPoseMask;
            cv::Mat opMaskedImage;
            std::vector<cv::Mat> histChannel;
            std::vector<cv::Mat> histGraph;
            cv::Point2f rFoot;
            cv::Point2f lFoot;
            cv::Mat rectMaskedIm;
            float outLineDist;
            cv::Point2f footInOV;
            cv::Point2f footInIm;
            bool ifSteps;
        };

        //Step
        class Step{
        public:
            cv::Point2f _foot;
            float _frame;
            float _pitch;
            float _stride;
            float _time;
        };

        std::vector<Step> meanStepList;
        std::vector<std::vector<OpenPoseBody>> _laneTrackingList;
        std::vector<std::vector<Step>> _laneStepList;
        std::vector<std::vector<Step>> _lanePitchList;
        std::vector<std::vector<Step>> _groundTruthList;
        std::vector<std::vector<float>> _lane10mSpeedList;
        std::vector<std::vector<float>> _laneModel10mSpeedList;
        std::vector<std::vector<float>> _lane10mtimeList;
        std::vector<std::vector<float>> _laneConstants;
        std::vector<int> _laneStepNum;
        std::vector<float> _goalTimeList;


        std::vector<Step> steps;

        //マスクエリア
        struct MaskArea {
            cv::Rect _mask_area;
        };

        std::vector<cv::Point2d> stepPoints;


        //Panorama
        cv::Mat OverView = cv::Mat::zeros(200, 1000, CV_8UC3);
        cv::Mat PanoramaImage = cv::Mat::zeros(20000, 40000, CV_8UC3);
        cv::Mat smallPanoramaImage;
        cv::Mat OriginalPanorama;
        cv::Mat overviewPanorama;
        cv::Mat inv_overView_H;
        cv::Mat affineH;

        int finalLineImageNum;

        std::vector<cv::Point2f> PanoramaLeftPoints;
        std::vector<cv::Point2f> PanoramaRightPoints;

        std::vector<cv::Point2f> panoramaInline10mPoints;
        std::vector<cv::Point2f> panoramaOutline10mPoints;


        std::vector<cv::Point2f> startLineCornerPoints;
        std::vector<cv::Point2f> finishLineCornerPoints;

        std::vector<cv::Point2f> overviewRightLegs;
        std::vector<cv::Point2f> overviewLeftLegs;

        //多項式近似ように歩幅をcv::Point2dベクトルに収納
        std::vector<float> strideLength;
        std::vector<cv::Point2d> stridePoints;

        float averagePitch;


        cv::Mat strobo_image = cv::Mat::zeros(8000, 30000, CV_8UC3);
        cv::Mat small_strobo = cv::Mat::zeros(8000, 30000, CV_8UC3);

        int panorama_offset = 50;

        //?��p?��m?��?��?��}?��p
        class ImageInfo {
        public:

            //images
            cv::Mat image;
            cv::Mat gray_image;
            cv::Mat hsv_image;
            cv::Mat edge;
            cv::Mat edge_horizontal;
            cv::Mat panorama_scale_im;
            cv::Mat trackLineImage;
            cv::Mat trackLineAndOpenPoseImage;

            //mask
            cv::Mat maskimage;
            cv::Mat maskedrunner;
            cv::Mat denseMask;
            std::vector<MaskArea> maskAreas;

            //Homography
            cv::Point2f translation;
            std::vector<cv::Point2f> translationList;
            cv::Mat H;
            cv::Mat mulH;
            cv::Mat inverseH;
            cv::Mat mulInvH;
            std::vector<cv::Point2f> onlinePointList;

            //Runners
            cv::Mat runnerCandidatesImage;
            std::vector<OpenPoseBody> runnerCandidate;
            std::vector<std::pair<cv::Point2f, cv::Point2f>> lines10m;
            std::vector<OpenPoseBody> Runners;

            //接地判定
            cv::Point2f originalRfoot;
            cv::Point2f originalLfoot;
            cv::Point2f panoramaRfoot;
            cv::Point2f panoramaLfoot;
            cv::Point2f overviewRfoot;
            cv::Point2f overviewLfoot;
            float RlineDist;
            float LlineDist;
            bool Rstep = false;
            bool Lstep = false;
            bool stepPoint = false;
            int frameID;

            //Strobo
            cv::Mat stroboH;
            cv::Mat strobo_scale_im;


            //?��}?��b?��`?��?��?��O?��p
            std::vector<cv::Point2f> prev_keypoints;
            std::vector<cv::Point2f> this_keypoints;


            //?��g?��?��?��b?��N?��?��?��̃g?��?��?��b?��L?��?��?��O
            std::vector<std::pair<cv::Point2f, cv::Point2f>> track_lines;
            std::vector<cv::Mat> track_line_masks;
            cv::Mat trackAreaMask;
            std::vector<float> grads;
            std::vector<float> segments;

        };

        cv::Mat overView_H;

        std::vector<ImageInfo> imList;

        int TARGET_RUNNER_ID;
        int TRACKING_MIN_DIST;
        int STEP_JUDGE_RANGE;


        void showOnlinePoints(ImageInfo &im);

        void obtainOnlinePointsAsIm1(ImageInfo& im, cv::Point2f translation);
        void obtainOnlinePointsAsIm2(ImageInfo& im);
        void myTemplateMatching(ImageInfo &im, ImageInfo &preim);

    private:
        //Panorama
        float Panorama_width;
        float Panorama_height;

        //smallPanorama
        float smallPanorama_width = 1000.0;
        float smallPanorama_height = 500.0;

        std::vector<MaskArea> mask_areas;
        std::vector<std::vector<OpenPoseBody> > allRunners;

    };
}

class step{
public:
    cv::Point2f pt;
    std::string s;
    int frame;
};