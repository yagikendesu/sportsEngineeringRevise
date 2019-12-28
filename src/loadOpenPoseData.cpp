//
// Created by yagi on 18/07/18.
//

#include "panorama.h"
#include "basicFunctions/basicFunction.h"
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace yagi;
using namespace cv;

//最小座標を求める
cv::Point minPoint(vector<cv::Point> contours){
    double minx = contours.at(0).x;
    double miny = contours.at(0).y;
    for(int i=1;i<contours.size(); i++){
        if(minx > contours.at(i).x){
            minx = contours.at(i).x;
        }
        if(miny > contours.at(i).y){
            miny = contours.at(i).y;
        }
    }
    return cv::Point(minx, miny);
}


//最大座標を求める
cv::Point maxPoint(vector<cv::Point> contours){
    double maxx = contours.at(0).x;
    double maxy = contours.at(0).y;
    for(int i=1;i<contours.size(); i++){
        if(maxx < contours.at(i).x){
            maxx = contours.at(i).x;
        }
        if(maxy < contours.at(i).y){
            maxy = contours.at(i).y;
        }
    }
    return cv::Point(maxx, maxy);
}

cv::Rect calcRectOfMask(cv::Mat mask){
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);

    int maxSize = 0;
    int maxID = 0;
    for(int i = 0; i < contours.size(); i++){
        if(maxSize < contours[i].size()){
            maxSize = int(contours[i].size());
            maxID = i;
        }
    }
    cv::Point minP = minPoint(contours.at(maxID) );
    cv::Point maxP = maxPoint(contours.at(maxID) );
    cv::Rect resultRect(minP, maxP);
    return resultRect;
}

//OpenPoseのデータ読みこみ
void Panorama::detectHumanArea() {
    cout << "[Detect human area]" << endl;
    ifstream ifs(_openpose_list_path);
    string line;
    OpenPoseBody runner;
//    vector<OpenPoseBody> runners;
    int frameID = 0;
    while (getline(ifs, line)) {
        vector<string> coords = split(line, ' ');
        if (coords[0] == "Frame:") {
            frameID++;
        }else if(coords[0] == "Person"){
            runner.clearBodyCoord();
        }else if(coords[0] == "end"){
            this->imList[frameID - 1].Runners.push_back(runner);
        }else{
            cv::Point2f pt(stof(coords[0]), stof(coords[1]));
            float conf = stof(coords[2]);
            runner.bodyPts.push_back(pt);
            runner._confidenceMap.push_back(conf);
        }
    }

//
//    OpenPoseBody runner;
//    vector<OpenPoseBody> runners;
//    vector<cv::Point2f> pre_centers;
//
//    bool first_runner = true;
//    int frame_counter = 0;
//
//    while (getline(ifs, line)) {
//
//        vector<string> coords = split(line, ' ');
//
//        if (coords.size() == 5) {
//
//            if (!first_runner) {
//
//                OpenPoseBody dummy_runner = runner;
//                runners.push_back(dummy_runner);
//                runner.clearBodyCoord();
//
//                if (coords[1] == "0") {
//                    vector<OpenPoseBody> dummy_runners = runners;
//                    allRunners.push_back(dummy_runners);
//
//                    runners.clear();
//                    frame_counter++;
//                }
//            }
//        } else {
//            runner.setBodyCoord(coords);
//            runner.mask_rect = getMaskRect(runner.bodyPts);
//            first_runner = false;
//        }
//
//        if (frame_counter == imList.size()) {
//            break;
//        }
//    }
//
//    for (int i = 0; i < allRunners.size(); i++) {
//        imList[i].Runners = allRunners[i];
//    }

    cout << "[Detect human area finished]" << endl;
}


void Panorama::OpenPoseBody::setBodyCoord(vector<string> coord) {
    cv::Point2f coord_f(stof(coord[0]), stof(coord[1]));
    bodyPts.push_back(coord_f);
}


vector<cv::Point2f> Panorama::OpenPoseBody::getBodyCoord() {
    return this->bodyPts;
}


void Panorama::OpenPoseBody::clearBodyCoord() {
    bodyPts.clear();
}

cv::Mat Panorama::calcOpenPoseMask(vector<cv::Point2f>& pts, cv::Size imSize){
    cv::Mat mask = cv::Mat::zeros(imSize, CV_8U);
    cv::Scalar WHITE(255,255,255);
    for(cv::Point2f pt: pts){
        if(pt.x != 0.0) {
            cv::circle(mask, pt, OP_MASK_RADIUS, WHITE, -1, CV_AA);
        }
    }
    cv::morphologyEx(mask, mask, MORPH_CLOSE, 2);
    return mask;
}

cv::Mat myGraphCut(cv::Mat& image, cv::Mat& result, cv::Rect rectangle, const int itrNum){

//    cv::Mat result; // segmentation result (4 possible values)
    cv::Mat bgModel,fgModel; // the models (internally used)

    // GrabCut segmentation
    cv::grabCut(image,    // input image
                result,   // segmentation result
                rectangle,// rectangle containing foreground
                bgModel,fgModel, // models
                itrNum,        // number of iterations
                cv::GC_INIT_WITH_RECT); // use rectangle

    // Get the pixels marked as likely foreground
    cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);

    // Generate output image
    cv::Mat foreground(image.size(),CV_8UC3,cv::Scalar(0,0,0));
    image.copyTo(foreground,result); // bg pixels not copied
//
    // draw rectangle on original image
    cv::rectangle(image, rectangle, cv::Scalar(255,255,255),1);
    cv::namedWindow("Image");
    cv::imshow("Image",image);

    // display result
    cv::namedWindow("Segmented Image");
    cv::imshow("Segmented Image",foreground);
    cv::waitKey();

    return foreground;
}
//
////最小座標を求める
//cv::Point minPoint(vector<cv::Point> contours){
//    double minx = contours.at(0).x;
//    double miny = contours.at(0).y;
//    for(int i=1;i<contours.size(); i++){
//        if(minx > contours.at(i).x){
//            minx = contours.at(i).x;
//        }
//        if(miny > contours.at(i).y){
//            miny = contours.at(i).y;
//        }
//    }
//    return cv::Point(minx, miny);
//}
////最大座標を求める
//cv::Point maxPoint(vector<cv::Point> contours){
//    double maxx = contours.at(0).x;
//    double maxy = contours.at(0).y;
//    for(int i=1;i<contours.size(); i++){
//        if(maxx < contours.at(i).x){
//            maxx = contours.at(i).x;
//        }
//        if(maxy < contours.at(i).y){
//            maxy = contours.at(i).y;
//        }
//    }
//    return cv::Point(maxx, maxy);
//}

void Panorama::getOpenPoseMask(){
    for(int imID = 0; imID < imList.size(); imID++){
        vector<OpenPoseBody>& ops = imList[imID].Runners;
        for(int opID = 0; opID < ops.size(); opID++){
            OpenPoseBody& hb = ops[opID];
            cv::Mat openPoseMask = calcOpenPoseMask(hb.bodyPts, cv::Size(IMG_WIDTH, IMG_HEIGHT));
            hb.openPoseMask = openPoseMask;
            cv::Mat roughMasked = yagi::maskAofB(imList[imID].image, hb.openPoseMask);
            hb.opMaskedImage = roughMasked;
            hb.mask_rect = calcRectOfMask(openPoseMask);
//            cv::imshow("aa", imList[imID].maskimage );

            cv::Mat img_not;
            bitwise_not(openPoseMask, img_not);
            imList[imID].maskimage = maskAofB( img_not, imList[imID].maskimage);

            //トラックラインマスク
            for(int i = 0; i < imList[imID].track_line_masks.size(); i++) {
                cv::Mat img_not;
                bitwise_not(imList[imID].track_line_masks[i], img_not);
                imList[imID].maskimage = maskAofB( img_not, imList[imID].maskimage);
            }


            cv::Mat rectMaskedIm(imList[imID].image, hb.mask_rect);
            cv::Mat rectMask(openPoseMask, hb.mask_rect);
            cv::resize(rectMaskedIm, rectMaskedIm, cv::Size(), 100.0/rectMaskedIm.cols, 100.0/rectMaskedIm.rows);
            cv::resize(rectMask, rectMask, cv::Size(), 100.0/rectMask.cols, 100.0/rectMask.rows);
            hb.rectMaskedIm = rectMaskedIm;

            //test トラッキングのために色ヒストグラムを取得
            cv::Mat histR, histG, histB, frame;
            cv::Mat channel[3];
            cv::split(rectMaskedIm, channel);

            cv::Mat graphR = drawHist(channel[0], rectMaskedIm, rectMask, histR);
            cv::Mat graphG = drawHist(channel[1], rectMaskedIm, rectMask, histG);
            cv::Mat graphB = drawHist(channel[2], rectMaskedIm, rectMask, histB);
            hb.histChannel.push_back(histR);
            hb.histChannel.push_back(histG);
            hb.histChannel.push_back(histB);
            hb.histGraph.push_back(graphR);
            hb.histGraph.push_back(graphG);
            hb.histGraph.push_back(graphB);

////        test トラッキングのために足関節と直線の距離を算出
            int ptID = 0;
            cv::Point2f rFoot(0,0);
            int rFootNum = 0;
            cv::Point2f lFoot(0,0);
            int lFootNum = 0;

            for(cv::Point2f pt: hb.bodyPts) {
                //右足左足の重心を出す
                if(pt != cv::Point2f(0,0)) {
                    if ((ptID == 19) || (ptID == 20) || (ptID == 21)) {
                        rFoot.x += pt.x;
                        rFoot.y += pt.y;
                        rFootNum++;
                    } else if (ptID == 22 || ptID == 23 || ptID == 24) {
                        lFoot.x += pt.x;
                        lFoot.y += pt.y;
                        lFootNum++;
                    }
                }
                ptID++;
            }
            rFoot.x/=rFootNum;
            lFoot.x/=lFootNum;
            rFoot.y/=rFootNum;
            lFoot.y/=lFootNum;
//            cv::circle(imList[imID].image, rFoot, 2, cv::Scalar(255,0,0), 2);
//            cv::circle(imList[imID].image, lFoot, 2, cv::Scalar(255,0,0), 2);
//            cv::imshow("a", imList[imID].image);
//            cv::waitKey();

            hb.rFoot = rFoot;
            hb.lFoot = lFoot;
//            float rdist = yagi::distPoint2Line(rFoot, imList[imID].grads[0], imList[imID].segments[0]);
//            float ldist = yagi::distPoint2Line(lFoot, imList[imID].grads[0], imList[imID].segments[0]);
//            cv::putText(imList[imID].image, to_string(int(dist)), pt, 0.5, 0.5, cv::Scalar(255, 0, 0), 2);

        }
//        cv::Mat maskIm = maskAofB(imList[imID].image, );

//        cv::imshow("a", imList[imID].maskimage );
//        cv::waitKey();

    }
}



void Panorama::maskingRunners(){
    for(ImageInfo im : imList){
        for(OpenPoseBody hb: im.runnerCandidate){
//            cv::Mat roughMasked = yagi::maskAofB(im.image, hb.openPoseMask);
//            cv::imshow("rough", roughMasked);
//            cv::Rect rect = calcRectOfMask(hb.openPoseMask);
//            cv::rectangle(roughMasked, rect, cv::Scalar(255,255,255), 2);
//            cv::imshow("rect", roughMasked);
//            cv::Mat maskedIm = myGraphCut(roughMasked, hb.maskedImage, rect, 2);
//            cv::imshow("masked", maskedIm);
//            cv::waitKey();
        }

    }
}

void Panorama::maskHumanArea() {
    cout << "[Mask human area]" << endl;

    for (auto itr_frame = imList.begin(); itr_frame != imList.end(); ++itr_frame) {
//        std::vector<OpenPoseBody> runnersInFrame = itr_frame->Runners;
        std::vector<MaskArea> mask_in_frame;
//
//        for (auto itr_runner = runnersInFrame.begin(); itr_runner != runnersInFrame.end(); ++itr_runner) {
//            MaskArea area;
//            area._mask_area = getMaskRect(itr_runner->bodyPts);
//            mask_in_frame.push_back(area);
//        }

        for (auto itr = mask_areas.begin(); itr != mask_areas.end(); ++itr) {
            mask_in_frame.push_back(*itr);
        }

        itr_frame->maskAreas = mask_in_frame;
    }



    //デバッグ マスク領域を表示

    for (auto itr_frame = imList.begin(); itr_frame != imList.end(); ++itr_frame) {

        cv::Scalar color(0, 0, 0);
        cv::Mat image = itr_frame->image;
        cv::Mat maskimage = cv::Mat::ones(itr_frame->image.size(), CV_8U) * 255;
        cv::Mat mask = image.clone();

        for (auto itr = itr_frame->maskAreas.begin(); itr != itr_frame->maskAreas.end(); ++itr) {
            cv::rectangle(mask, itr->_mask_area, color, CV_FILLED);
            cv::rectangle(maskimage, itr->_mask_area, color, CV_FILLED);
        }

        itr_frame->maskimage = maskimage;
        itr_frame->maskedrunner = mask;

        if (SHOW_MASK_REGIONS) {
            cv::imshow("mask", mask);
            cv::waitKey(0);
        }
    }

    cout << "[Mask human area finished]" << endl;
}


cv::Rect Panorama::getMaskRect(vector<cv::Point2f> &bodyPts) {
    int min_x = 10000;
    int max_x = 0;
    int min_y = 10000;
    int max_y = 0;
    for (auto itr_coord = bodyPts.begin(); itr_coord != bodyPts.end(); ++itr_coord) {
        int x = itr_coord->x;
        int y = itr_coord->y;
        if (min_x > x)
            if (x > 0)
                min_x = x;
        if (min_y > y)
            if (y > 0)
                min_y = y;
        if (max_x < x)
            max_x = x;
        if (max_y < y)
            max_y = y;
    }

    // マスク領域の指定
//    int margin_x_left = 50;
//    int margin_x_right = 50;
//    int margin_y_up = 50;
//    int margin_y_down = 80;
    return cv::Rect(min_x - MASK_MARGIN, min_y - MASK_MARGIN, max_x - min_x + MASK_MARGIN,
                    max_y - min_y + MASK_MARGIN);
}
