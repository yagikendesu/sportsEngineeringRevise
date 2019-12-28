//
// Created by yagi on 18/07/18.
//

#include "./panorama.h"
#include "basicFunctions/basicFunction.h"

using namespace yagi;
using namespace std;

void siftMatching(cv::Mat &im1, cv::Mat &im2){

    //アルゴリズムにAKAZEを使用する
    auto algorithm = cv::AKAZE::create();

    // 特徴点抽出
    std::vector<cv::KeyPoint> keypoint1, keypoint2;
    algorithm->detect(im1, keypoint1);
    algorithm->detect(im2, keypoint2);

    // 特徴記述
    cv::Mat descriptor1, descriptor2;
    algorithm->compute(im1, keypoint1, descriptor1);
    algorithm->compute(im2, keypoint2, descriptor2);

    // マッチング (アルゴリズムにはBruteForceを使用)
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
    std::vector<cv::DMatch> match, match12, match21;
    matcher->match(descriptor1, descriptor2, match12);
    matcher->match(descriptor2, descriptor1, match21);
    //クロスチェック(1→2と2→1の両方でマッチしたものだけを残して精度を高める)
    for (size_t i = 0; i < match12.size(); i++)
    {
        cv::DMatch forward = match12[i];
        cv::DMatch backward = match21[forward.trainIdx];
        if (backward.trainIdx == forward.queryIdx)
        {
            match.push_back(forward);
        }
    }

    // マッチング結果の描画
    cv::Mat dest;
    cv::drawMatches(im1, keypoint1, im2, keypoint2, match, dest);
    cv::imshow("matching result", dest);
    cv::waitKey();

}

void Panorama::featurePointFindHomography(){
    cv::Mat im1, im2;
    int frameNum = 0;
    for(ImageInfo im: this->imList){
        im1 = im.image;
        if(frameNum > 0){
            siftMatching(im1, im2);
        }
        im2 = im1;
        frameNum++;
    }
}

cv::Mat getHscaleImage(cv::Mat& im) {
    vector<cv::Mat> chunnels;
    cv::split(im, chunnels);
    return chunnels[0];
}

cv::Point2f Panorama::templateMatching(cv::Mat& im1, cv::Mat& im2, cv::Rect tempRect, cv::Mat& maskImage, cv::Point2f preTranslation, const int frameID) {
    cv::Mat templateImg(im1, tempRect);
    cv::Mat templateMask(maskImage, tempRect);
    cv::imshow("a", templateImg);
    cv::imshow("aaa", templateMask);

    //テンプレートマッチング
    cv::Mat tempResult;
    cv::matchTemplate(im2, templateImg, tempResult, CV_TM_CCORR_NORMED, templateMask);

    // 最大のスコアの場所を探す
    cv::Rect roi_rect(0, 0, templateImg.cols, templateImg.rows);
    cv::Point max_pt;
    double maxVal;
    cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt);

    //Point型からPoint2f型に変換するため
    cv::Point2f translation;
    translation.x = max_pt.x;
    translation.y = max_pt.y;

    //前のフレーム(pre_pt)とトランスレーションが大きく異なる場合には第二最大値を選択
    cv::Mat mask = cv::Mat::ones(tempResult.size(), CV_8U) * 255;
    for(int i = 0; i < 1; i++ ) {
        cv::Point2f translation;
        cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt, mask);
        translation.x = max_pt.x;
        translation.y = max_pt.y;
        while ((calc2PointDistance(translation, preTranslation) > MAX_TRANSLATION)) {
            mask.at<unsigned char>(max_pt) = 0;
            cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt, mask);
            translation.x = max_pt.x;
            translation.y = max_pt.y;
        }
        imList[frameID].translationList.push_back(translation);
        mask.at<unsigned char>(max_pt) = 0;
    }
    cv::waitKey(1);
    return imList[frameID].translationList[0];
}

void Panorama::showOnlinePoints(ImageInfo& im){
    cv::Mat image = im.image.clone();
    for(cv::Point2f pt: im.this_keypoints){
        cv::circle(image, pt, 2, cv::Scalar(255,0,0), 2);
    }
    for(cv::Point2f pt: im.prev_keypoints){
        cv::circle(image, pt, 2, cv::Scalar(0,0,255), 2);
    }
    cv::imshow("onLinePoints", image);
    cv::waitKey();
}

void Panorama::getTranslationByMyTempMatching() {
    string tmpImPath = _result_folder + "/template/";
    myMkdir(tmpImPath);
    string bestPath = tmpImPath + "/bestTemp/";
    myMkdir(bestPath);

    for (int frameID = 0; frameID < imList.size(); frameID++) {
        if(frameID == 0){
            imList[frameID].translation = cv::Point2f(0,0);
        }else {
            myTemplateMatching(imList[frameID], imList[frameID - 1]);
        }
    }
}

void Panorama::getTranslationByTempMatching() {

    cout << "[calculate translation by template matching]" << endl;
    ofstream translationTxt(_txt_folder + "/translation.txt");
    int sumTranslation = 0;

    //テンプレートマッチ用のrect size指定
    cv::Point2f start_pt(0, 0);
    int width = 550;
    int height = 280;
    cv::Rect tempRect(int(start_pt.x), int(start_pt.y), width, height);

    int frame_num = 0;
    cv::Mat im1, im2;
    cv::Point2f preTranslation;

    for (int frameID = 0; frameID < imList.size(); frameID++) {
        im1 = imList[frameID].hsv_image;
        cv::Point2f translation;
        if(frameID == 0){
            translation = cv::Point2f(0,0);
        }else {
            //hsvスケール
            cv::Mat channel[3];
            cv::split(im1, channel);
            cv::Mat im1H = channel[0].clone();

            cv::split(im2, channel);
            cv::Mat im2H = channel[0].clone();

//            cv::Mat im1H = yagi::maskAofB(im1, imList[frameID].trackAreaMask);
//            cv::Mat im2H = yagi::maskAofB(im2, imList[frameID].trackAreaMask);
            cv::imshow("a", im1H);
            cv::imshow("b", im2H);
            cv::waitKey();

//            cv::bitwise_and(imList[frameID].maskimage, imList[frameID].trackAreaMask, imList[frameID].maskimage);
//            cv::imshow("b", imList[frameID].maskimage);
//            translation = templateMatching(im1, im2, tempRect, imList[frameID].maskimage, preTranslation, frameID);
        }
        frame_num++;
        im2 = im1;
        preTranslation = translation;
        imList[frameID].translation = translation;
        sumTranslation += translation.x;

        translationTxt << translation.x << endl;
    }
    cv::destroyAllWindows();
    cout << "[calculate translation by template matching finished]" << endl;
}

void Panorama::getTranslationByBatchTempMatching() {

    cout << "[calculate translation by template matching]" << endl;

    const int BATCH_SIDE_LEN = 3;
    const int TEMPRECT_MARGIN = 5;
    //テンプレートマッチ用のrect size指定
    cv::Point2f start_pt(0, 0);
    int WIDTH = imList[0].image.cols;
    int HEIGHT = imList[0].image.rows;
    int batchWidth = imList[0].image.cols / BATCH_SIDE_LEN;
    int batchHeight = imList[0].image.rows / BATCH_SIDE_LEN;

    cv::Mat im1H;
    cv::Mat im2H;
    cv::Mat im1, im2;
    for (int frameID = 0; frameID < imList.size(); frameID++) {
        im1 = imList[frameID].hsv_image;
        cv::Point2f maxTranslation;
        if (frameID == 0) {
            maxTranslation = cv::Point2f(0, 0);
        } else {
            for(int i = 0; i < im2.rows; i++){
                for(int j = 0; j < im2.cols; j++){
                    if(imList[frameID].maskimage.at<unsigned char>(i,j) == 0)
                        im2.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
                }
            }
            cv::imshow("im2", im2);
            im1H = getHscaleImage(im1);
            im2H = getHscaleImage(im2);
//            cv::Mat im1H = imList[frameID].gray_image;
//            cv::Mat im2H = imList[frameID - 1].gray_image;

            cv::imshow("im1H", im1H);
            cv::imshow("im2H", im2H);
            cv::imshow("mask", imList[frameID].maskimage);

            double maxScore = 0;
            cv::Mat maxTemp;
            cv::Mat maxBatch;
            for (int i = 0; i < BATCH_SIDE_LEN; i++) {
                for (int j = 0; j < BATCH_SIDE_LEN; j++) {
                    cv::Rect batchRect(int(start_pt.x + (i * batchWidth)),
                                       int(start_pt.y + (j * batchHeight)),
                                       batchWidth, batchHeight);
                    cv::Rect tempRect(int(start_pt.x + (i * batchWidth) + TEMPRECT_MARGIN),
                                      int(start_pt.y + (j * batchHeight) + TEMPRECT_MARGIN),
                                      batchWidth - 2 * TEMPRECT_MARGIN,
                                      batchHeight - 2 * TEMPRECT_MARGIN);

                    cv::Mat templateImg(im1H, tempRect);
                    cv::Mat baseImg(im2H, batchRect);



//
//                cv::Mat templateMask(maskImage, tempRect);

                    //テンプレートマッチング
                    cv::Mat tempResult;
                    cv::matchTemplate(baseImg, templateImg, tempResult, CV_TM_CCORR_NORMED);
//                cv::matchTemplate(im2H, templateImg, tempResult, CV_TM_CCORR_NORMED, templateMask);

                    // 最大のスコアの場所を探す
                    cv::Point max_pt;
                    double maxVal;
                    cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt);
                    if (maxVal > maxScore) {
                        maxScore = maxVal;
                        maxTranslation.x = max_pt.x;
                        maxTranslation.y = max_pt.y;
                        maxTemp = templateImg;
                        maxBatch = baseImg;
                    }
                }
            }
            int x = int(maxTranslation.x - TEMPRECT_MARGIN);
            int y = int(maxTranslation.y - TEMPRECT_MARGIN);

            //TranslationのH求める
            vector<cv::Point2f> prevPts;
            prevPts.push_back(cv::Point2f(0, 0));
            prevPts.push_back(cv::Point2f(WIDTH, 0));
            prevPts.push_back(cv::Point2f(0, HEIGHT));
            prevPts.push_back(cv::Point2f(WIDTH, HEIGHT));

            vector<cv::Point2f> nextPts;
            nextPts.push_back(cv::Point2f(x, y));
            nextPts.push_back(cv::Point2f(WIDTH + x, y));
            nextPts.push_back(cv::Point2f(x, HEIGHT + y));
            nextPts.push_back(cv::Point2f(WIDTH + x, HEIGHT + y));

            if (x < 0)
                x = 0;
            if (y < 0)
                y = 0;

            cv::Mat transH = cv::findHomography(prevPts, nextPts);
            cv::Mat warpedSrc;
            cv::warpPerspective(imList[frameID].image, warpedSrc, transH,
                                cv::Size(WIDTH + x, HEIGHT + y));

            for (int x = 0; x < HEIGHT; x++) {
                for (int y = 0; y < WIDTH; y++) {
                    if (warpedSrc.at<cv::Vec3b>(x, y) == cv::Vec3b(0, 0, 0)) {
                        warpedSrc.at<cv::Vec3b>(x, y) = imList[frameID - 1].image.at<cv::Vec3b>(x, y);
                    }
                }
            }
            cout << maxScore << " " << maxTranslation << endl;
            cv::imshow("temp", maxTemp);
            cv::imshow("base", maxBatch);
            cv::imshow("Translation Result", warpedSrc);
            cv::waitKey();
        }
        imList[frameID].translationList.push_back(maxTranslation);
        im2 = im1;
    }
    cv::destroyAllWindows();
    cout << "[calculate translation by template matching finished]" << endl;
}


//                //前のフレーム(pre_pt)とトランスレーションが大きく異なる場合には第二最大値を選択
//                cv::Mat mask = cv::Mat::ones(tempResult.size(), CV_8U) * 255;
//                for(int i = 0; i < 5; i++ ) {
//                    cv::Point2f translation;
//                    cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt, mask);
//                    translation.x = max_pt.x;
//                    translation.y = max_pt.y;
//                    while ((calc2PointDistance(translation, preTranslation) > MAX_TRANSLATION)) {
//                        cout << frameID << endl;
//                        mask.at<unsigned char>(max_pt) = 0;
//                        cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt, mask);
//                        translation.x = max_pt.x;
//                        translation.y = max_pt.y;
//                    }
//                    imList[frameID].translationList.push_back(translation);
//                    mask.at<unsigned char>(max_pt) = 0;






void Panorama::obtainOnlinePointsAsIm1(ImageInfo& im, cv::Point2f translation){
    CV_Assert(im.grads.size() == im.segments.size());
    for (int ptID = 0; ptID < im.grads.size(); ptID++) {
        float a = im.grads[ptID];
        float b = im.segments[ptID];

        cv::Point2f prev1(translation.x, a * (translation.x) + b);
        cv::Point2f prev2(im.image.cols, (a * im.image.cols) + b);

        im.prev_keypoints.push_back(prev1);
        im.prev_keypoints.push_back(prev2);
    }
}

void Panorama::obtainOnlinePointsAsIm2(ImageInfo& im){
    CV_Assert(im.grads.size() == im.segments.size());
    for (int ptID = 0; ptID < im.grads.size(); ptID++) {
        float a = im.grads[ptID];
        float b = im.segments[ptID];

        cv::Point2f this1(0, b);
        cv::Point2f this2(im.image.cols - (im.translation.x),
                          a * (im.image.cols - (im.translation.x )) + b);

        im.Rstep = true;
        im.this_keypoints.push_back(this1);
        im.this_keypoints.push_back(this2);
    }
}

void showHomographyWarpingResult(cv::Mat& im1, cv::Mat &im2, cv::Mat H){
    int WARPING_BLANK = 100;

    cv::Mat warped;
    cv::warpPerspective(im1, warped, H,
                        cv::Size(im1.cols + WARPING_BLANK, im1.rows + WARPING_BLANK));

    for (int x = 0; x < im1.rows; x++) {
        for (int y = 0; y <im1.cols; y++) {
            if (warped.at<cv::Vec3b>(x, y) == cv::Vec3b(0,0,0)) {
                warped.at<cv::Vec3b>(x, y) = im2.at<cv::Vec3b>(x, y);
            }
        }
    }
    cv::imshow("Homography Result", warped);
    cv::waitKey();
}


void Panorama::getHomographyFromTranslation(){
    cv::Mat im1, im2, H;
    ImageInfo preIm;

    for (int frameID = 0; frameID < imList.size(); frameID++) {
        ImageInfo& im = imList[frameID];
        im1 = im.image;
        if(frameID == 0){
            cv::Mat initH = cv::Mat::zeros(3, 3, CV_64F);
            initH.at<double>(0, 0) = 1;
            initH.at<double>(1, 1) = 1;
            initH.at<double>(2, 2) = 1;
            H = initH;
            im2 = im1;
        }else {
            ImageInfo& preIm = imList[frameID - 1];
            obtainOnlinePointsAsIm1(preIm, im.translation);
            obtainOnlinePointsAsIm2(im);
            if(SHOW_ONLINE_POINTS)
                showOnlinePoints(im);
            H = cv::findHomography(im.this_keypoints,preIm.prev_keypoints,
                                                   CV_RANSAC, 1.0);
        }
        imList[frameID].H = H;
        if(SHOW_HOMOGRAPHY) {
            cout << imList[frameID].translation << endl;
            showHomographyWarpingResult(im1, im2, H);
        }
        im2 = im1;
    }
}

//void Panorama::getTranslationByTempMatching() {
//
//    cout << "[calculate translation by template matching]" << endl;
//    ofstream outTxt("./translation.txt");
//
//    //テンプレートマッチ用のrect size指定
//    cv::Point2f start_pt(40, 20);
//    int width = 550;
//    int height = 280;
//    cv::Rect tempRect(start_pt.x, start_pt.y, width, height);
//    cv::Mat im1, im2;
//    cv::Point2f pre_pt = start_pt;
//
//    //ホモグラフィー用の点格納
//    vector<cv::Point2f> this_points;
//    vector<cv::Point2f> next_points;
//
//    int frame_num = 0;
//    for (int frameID = 0; frameID < imList.size(); frameID++) {
//        if(frameID == 0){
//            im1 = imList[frameID].image;
//            imList[frameID].translation = cv::Point2f(0,0);
////            cv::Mat initH = cv::Mat::zeros(3, 3, CV_64F);
////            initH.at<double>(0, 0) = 1;
////            initH.at<double>(1, 1) = 1;
////            initH.at<double>(2, 2) = 1;
////            imList[0].H = initH.clone();
//        }else if(frameID > 0) {
//            //このフレーム画像
//            cv::Mat prevHsvImg = imList[frameID - 1].hsv_image;
//            cv::Mat thisHsvImg = imList[frameID].hsv_image;
//
////            cv::imshow("a", prevHsvImg);
////            cv::imshow("b", thisHsvImg);
//            vector<cv::Mat> prev_planes;
//            vector<cv::Mat> this_planes;
//            cv::split(prevHsvImg, prev_planes);
//            cv::split(thisHsvImg, this_planes);
//
//            cv::Mat prevImg = prev_planes[0];
//            cv::Mat thisImg = this_planes[0];
//
//            cv::Mat templateImg(thisImg, tempRect);
//            cv::Mat tempMask(imList[frameID].maskimage, tempRect);
//
//            cv::Mat maskedTemp = tempImg.clone();
//            for (int i = 0; i < tempMask.rows; i++) {
//                for (int j = 0; j < tempMask.cols; j++) {
//                    if (tempMask.at<unsigned char>(i, j) == 0) {
//                        maskedTemp.at<unsigned char>(i, j) = 0;
//                    }
//                }
//            }
////        cv::imshow("dummy mask area", maskedTemp);
//
//            //テンプレートマッチング
//            cv::Mat tempResult;
//            cv::matchTemplate(prevImg, tempImg, tempResult, CV_TM_CCORR_NORMED, tempMask);
//
//            // 最大のスコアの場所を探す
//            cv::Rect roi_rect(0, 0, tempImg.cols, tempImg.rows);
//            cv::Point max_pt;
//            double maxVal;
//            cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt);
//
//            //Point型からPoint2f型に変換するため
//            cv::Point2f max_pt2f;
//            max_pt2f.x = max_pt.x;
//            max_pt2f.y = max_pt.y;
//
//            //前のフレーム(pre_pt)とトランスレーションが大きく異なる場合には第二最大値を選択
//            if (!first_frame) {
//                cv::Mat mask = cv::Mat::ones(tempResult.size(), CV_8U) * 255;
//                while ((calc2PointDistance(max_pt2f, pre_pt) > 3) && (max_pt2f.y <= 40)) {
//                    mask.at<unsigned char>(max_pt) = 0;
//                    cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt, mask);
//                    max_pt2f.x = max_pt.x;
//                    max_pt2f.y = max_pt.y;
//                }
////                cout << frame_num << "PrePt " << pre_pt << ": maxPt " << max_pt << " "
////                     << calc2PointDistance(max_pt2f, pre_pt) << endl;
//                outTxt << frame_num << "PrePt " << pre_pt << ": maxPt " << max_pt << " "
//                       << calc2PointDistance(max_pt2f, pre_pt) << endl;
//            } else {
//                first_frame = false;
//            }
//
//            //pre_pt更新
//            pre_pt = max_pt;
//            roi_rect.x = max_pt.x;
//            roi_rect.y = max_pt.y;
//
//            //カラー画像でテンプレート最大値に対応領域を当て貼める
//            cv::Mat prevDummy = imList[frameID - 1].image.clone();
//            cv::Mat thisDummy = imList[frameID].image.clone();
//            cv::Mat tempDummy(thisDummy, tempRect);
//            cv::rectangle(prevDummy, roi_rect, cv::Scalar(255, 255, 255));
//
//            for (int x = max_pt.x; x < max_pt.x + width; x++) {
//                for (int y = max_pt.y; y < max_pt.y + height; y++) {
//                    prevDummy.at<cv::Vec3b>(y, x) = tempDummy.at<cv::Vec3b>(y - max_pt.y, x - max_pt.x);
//                }
//            }
//            cv::circle(prevDummy, max_pt, 2, cv::Scalar(255, 0, 0), 2);
//
////            cv::imshow("temp", prevDummy);
////            cv::waitKey();
//
//            //平行移動から直線上の4点をhomographyの対応点として選択
//            //prev frame用
//            for (int pt = 0; pt < imList[frameID - 1].grads.size(); pt++) {
//                float a1 = imList[frameID - 1].grads[pt];
//                float b1 = imList[frameID - 1].segments[pt];
//
//                cv::Point2f prev1(max_pt.x - start_pt.x, a1 * (max_pt.x - start_pt.x) + b1);
//                cv::Point2f prev2(prevImg.cols, (a1 * prevImg.cols) + b1);
//
//                imList[frameID].prev_keypoints.push_back(prev1);
//                imList[frameID].prev_keypoints.push_back(prev2);
//            }
//
//            //this frame用
//            for (int pt = 0; pt < imList[frameID].grads.size(); pt++) {
//                float a1 = imList[frameID].grads[pt];
//                float b1 = imList[frameID].segments[pt];
//
//                cv::Point2f this1(0, b1);
//                cv::Point2f this2(thisImg.cols - (max_pt.x - start_pt.x),
//                                  a1 * (thisImg.cols - (max_pt.x - start_pt.x)) + b1);
//
//                imList[frameID].this_keypoints.push_back(this1);
//                imList[frameID].this_keypoints.push_back(this2);
//
////                デバッグ
//                drawLine(imList[frameID].trackLineImage, imList[frameID].track_lines[pt].first, imList[frameID].track_lines[pt].second, 1, cv::Scalar(255,0,0));
////                cv::imshow("aa",imList[i].trackLineImage);
////                cv::waitKey();
//            }
//
//            //Debug：対応点を描画
//            cv::Mat prev_dum = imList[frameID - 1].trackLineImage.clone();
//            cv::Mat this_dum = imList[frameID].trackLineImage.clone();
//            cv::Scalar GREEN(0, 255, 0);
//            cv::Scalar BLUE(255, 0 , 0);
//
//            for (int pt = 0; pt < imList[frameID].prev_keypoints.size() - 1; pt++) {
//                cv::Point2f prev_pt = imList[frameID].prev_keypoints[pt];
//                cv::Point2f this_pt = imList[frameID].this_keypoints[pt];
//                cv::circle(prev_dum, prev_pt, 3, GREEN, 2);
//                cv::circle(this_dum, this_pt, 3, BLUE, 2);
//            }
////            cv::imshow("a", prev_dum);
////            cv::imshow("b", this_dum);
////            cv::waitKey();
//
//            imList[frameID].H = cv::findHomography(imList[frameID].this_keypoints,
//                                                   imList[frameID].prev_keypoints,
//                                                   CV_RANSAC, 1.0);
//        }
//        frame_num++;
//    }
//    cv::destroyAllWindows();
//    outTxt.close();
//    cout << "[calculate translation by template matching finished]" << endl;
//
//}


//void Panorama::templateInverseMatchingFindHomography() {
//
//    cout << "calculate translation from last frame by template matching" << endl;
//
//    cv::Mat thisImg = imList[0].image.clone();
//
//    //テンプレートマッチ用のrect size指定
//    cv::Point2f start_pt(40, 20);
//    int width = 550;
//    int height = 280;
//    cv::Rect tempRect(start_pt.x, start_pt.y, width, height);
//    cv::Point2f pre_pt = start_pt;
//
//    //ホモグラフィー用の点格納
//    vector<cv::Point2f> this_points;
//    vector<cv::Point2f> next_points;
//    bool first_frame = true;
//
//    //最後のフレームのホモグラフィー
//    cv::Mat mul_H = cv::Mat::zeros(3, 3, CV_64F);
//    mul_H.at<double>(0, 0) = 1;
//    mul_H.at<double>(1, 1) = 1;
//    mul_H.at<double>(2, 2) = 1;
//    imList[imList.size() - 1].inverseH = mul_H.clone();
//
//    for (int i = imList.size() - 2; i >= 0; i--) {
//
//        //このフレーム画像
//        cv::Mat thisHsvImg = imList[i].hsv_image;
//        cv::Mat nextHsvImg = imList[i + 1].hsv_image;
//        vector<cv::Mat> this_planes;
//        vector<cv::Mat> next_planes;
//        cv::split(thisHsvImg, this_planes);
//        cv::split(nextHsvImg, next_planes);
//
//        cv::Mat thisImg = this_planes[0];
//        cv::Mat nextImg = next_planes[0];
//
//        cv::Mat tempImg(nextImg, tempRect);
//        cv::Mat tempMask(imList[i].maskimage, tempRect);
//
//        //テンプレートマッチング
//        cv::Mat tempResult;
//        cv::matchTemplate(thisImg, tempImg, tempResult, CV_TM_CCORR_NORMED, tempMask);
//
//        // 最大のスコアの場所を探す
//        cv::Rect roi_rect(0, 0, tempImg.cols, tempImg.rows);
//        cv::Point max_pt;
//        double maxVal;
//        cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt);
//
//        //Point型からPoint2f型に変換するため
//        cv::Point2f max_pt2f;
//        max_pt2f.x = max_pt.x;
//        max_pt2f.y = max_pt.y;
//
//        //前のフレーム(pre_pt)とトランスレーションが大きく異なる場合には第二最大値を選択
//        if (!first_frame) {
//            cv::Mat mask = cv::Mat::ones(tempResult.size(), CV_8U) * 255;
//            while (calc2PointDistance(max_pt2f, pre_pt) > 3) {
//                mask.at<unsigned char>(max_pt) = 0;
//                cv::minMaxLoc(tempResult, NULL, &maxVal, NULL, &max_pt, mask);
//                max_pt2f.x = max_pt.x;
//                max_pt2f.y = max_pt.y;
//            }
//        } else {
//            first_frame = false;
//        }
//
//        //pre_pt更新
//        pre_pt = max_pt;
//        roi_rect.x = max_pt.x;
//        roi_rect.y = max_pt.y;
//
//        //カラー画像でテンプレート最大値に対応領域を当て貼める
//        cv::Mat thisDummy = imList[i].image.clone();
//        cv::Mat nextDummy = imList[i + 1].image.clone();
//        cv::Mat tempDummy(nextDummy, tempRect);
//        cv::rectangle(thisDummy, roi_rect, cv::Scalar(255, 255, 255));
//
//        for (int x = max_pt.x; x < max_pt.x + width; x++) {
//            for (int y = max_pt.y; y < max_pt.y + height; y++) {
//                thisDummy.at<cv::Vec3b>(y, x) = tempDummy.at<cv::Vec3b>(y - max_pt.y, x - max_pt.x);
//            }
//        }
//
//        //平行移動から直線上の4点をhomographyの対応点として選択
//        //this frame用
//        for (int pt = 1; pt < imList[i].grads.size() - 1; pt++) {
//            float a1 = imList[i].grads[pt];
//            float b1 = imList[i].segments[pt];
//
//            cv::Point2f this1(max_pt.x - start_pt.x, a1 * (max_pt.x - start_pt.x) + b1);
//            cv::Point2f this2(thisImg.cols, (a1 * thisImg.cols) + b1);
//
//            imList[i].this_keypoints.push_back(this1);
//            imList[i].this_keypoints.push_back(this2);
//        }
//
//        //next frame用
//        for (int pt = 1; pt < imList[i + 1].grads.size() - 1; pt++) {
//            float a1 = imList[i + 1].grads[pt];
//            float b1 = imList[i + 1].segments[pt];
//
//            cv::Point2f next1(0, b1);
//            cv::Point2f next2(nextImg.cols - (max_pt.x - start_pt.x),
//                              a1 * (nextImg.cols - (max_pt.x - start_pt.x)) + b1);
//
//            imList[i].next_keypoints.push_back(next1);
//            imList[i].next_keypoints.push_back(next2);
//        }
//
//        //Debug：対応点を描画
//        cv::Mat this_dum = imList[i].image.clone();
//        cv::Mat next_dum = imList[i + 1].image.clone();
//        cv::Scalar GREEN(0, 255, 0);
//
//        for (int pt = 0; pt < imList[i].this_keypoints.size() - 1; pt++) {
//            cv::Point2f this_pt = imList[i].this_keypoints[pt];
//            cv::Point2f next_pt = imList[i].next_keypoints[pt];
//            cv::circle(this_dum, this_pt, 3, GREEN, 2);
//            cv::circle(next_dum, next_pt, 3, GREEN, 2);
//        }
//
////        cv::imshow("t_dum", this_dum);
////        cv::imshow("n_dum", next_dum);
////        cv::waitKey();
//
//        imList[i].inverseH = cv::findHomography(imList[i].this_keypoints, imList[i].next_keypoints,
//                                                  CV_RANSAC, 3.0);
//
//    }
//    cv::destroyAllWindows();
//
//    cout << "calculate translation finished" << endl;
//}