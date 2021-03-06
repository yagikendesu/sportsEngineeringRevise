//
// Created by yagi on 18/01/10.
//

#include "panorama.h"
#include "basicFunctions/basicFunction.h"
#include "calcLiniorEquation.h"

using namespace std;
using namespace yagi;

//トラック領域をクリックで指定
//クリック用のグローバル変数
bool click_flag = false;
int mask_area_flag = 0;
cv::Point2f click_point;
vector<cv::Point2f> click_points;


//マウス入力用のパラメータ
struct mouseParam {
    int x;
    int y;
};


//プロトタイプ宣言
cv::Mat maskOutArea(cv::Mat in, cv::Mat out);
static void SelectTrackCallBackFunc(int eventType, int x, int y, int flags, void *userdata);


//コールバック関数
void SelectTrackCallBackFunc(int eventType, int x, int y, int flags, void *userdata) {
    switch (eventType) {
        case cv::EVENT_LBUTTONUP:
            std::cout << "[" << x << " , " << y << "] " << click_point << std::endl;
            click_point.x = x;
            click_point.y = y;
            click_points.push_back(click_point);
            click_flag = true;
            mask_area_flag++;
    }
}


void Panorama::selectTrack() {

    cout << "[Click track lines]" << endl;

    //カラー変数
    cv::Scalar WHITE(255, 255, 255);
    cv::Scalar BLUE(255, 0, 0);

    //クリック用の変数準備
    mouseParam mouseEvent;
    string window_name = "select track line by clicking 2 points";
    cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback(window_name, SelectTrackCallBackFunc, &mouseEvent);

    //最初のフレームを表示
    cv::Mat first_image = imList[0].image.clone();
    cv::imshow(window_name, first_image);

    //クリック点ファイル名前
    string click_file_name = _txt_folder + "/trackLineDetection.txt";

    //クリックするためのループ
    if (!checkFileExistence(click_file_name)) {
        ofstream outputfile(click_file_name);
        while (1) {
            int key = cv::waitKey(1);

            if (click_flag) {
                cv::circle(first_image, click_point, 2, BLUE, 2);
                cv::imshow(window_name, first_image);
                click_flag = false;
            }

            if (key == 'q') {
                for (cv::Point2f pt: click_points){
                    outputfile << pt.x << " " << pt.y << endl;
                }
                break;
            }
        }
        cv::destroyWindow(window_name);
    }else{
        //テキストファイルからクリック点読み込み
        std::ifstream ifs(click_file_name);
        std::string str;
        while(getline(ifs,str))
        {
//            std::cout<< str << std::endl;
            vector<string> point = split(str, ' ');
            cv::Point2f pt(stof(point[0]), stof(point[1]));
            click_points.push_back(pt);
        }
    }

    //クリック点をImageInfoに格納(2点ずつループ)
    cv::Mat dummy =imList[0].image.clone();
    for (auto itr = click_points.begin(); itr != click_points.end(); itr += 2) {
        float a, b;
        pair<cv::Point2f, cv::Point2f> line(*itr, *(itr + 1));
        yagi::getGradSegment(line.first, line.second, &a, &b);
        imList[0].track_lines.push_back(line);
        imList[0].grads.push_back(a);
        imList[0].segments.push_back(b);

        cv::Mat line_mask = cv::Mat::zeros(first_image.rows, first_image.cols, CV_8U);
        drawLine(line_mask, line.first, line.second, 5, WHITE);
        drawLine(dummy, line.first, line.second, 1, cv::Scalar(0,0,255));
        imList[0].track_line_masks.push_back(line_mask);
    }
//    cout << imList[0].grads.size() << endl;
    imList[0].trackLineImage = dummy;

    cout << "[Click track lines finished]" << endl;

}

//トラック領域のトラッキング
void Panorama::trackTracking() {

    //カラー変数
    cv::Scalar WHITE(255, 255, 255);
    cv::Scalar RED(0, 0, 255);

    cout << "[Tracking track line]" << endl;

    ofstream gradTxt(_txt_folder + "/grad.txt");

    //全画像のループ
    for (int i = 1; i < imList.size(); i++) {

        //今のフレームの画像
        cv::Mat edge = imList[i].edge;
        cv::Mat dummy = imList[i].image.clone();
        float gradAve = 0;
        //全レーンのループ
        for (int line = 0; line <  imList[i - 1].track_lines.size(); line++) {

            //今のフレームのedgeを前フレームのライン周辺でマスク
            cv::Mat line_neighbor = maskAofB(imList[i - 1].track_line_masks[line], edge);
//            cv::imshow("line", line_neighbor);
//            cv::waitKey();

            //RANSAC + 最小二乗法で各直線求めマスク
            cv::Point2f pt1, pt2;
            calcLine(line_neighbor, &pt1, &pt2, RANSAC_LOOP_LIMIT, RANSAC_INLIER_RANGE);
            pair<cv::Point2f, cv::Point2f> track_line(pt1, pt2);
            imList[i].track_lines.push_back(track_line);

            //求めた直線周辺をマスク
            cv::Mat line_mask = cv::Mat::zeros(edge.size(), CV_8U);
            drawLine(line_mask, pt1, pt2, 7, WHITE);
            imList[i].track_line_masks.push_back(line_mask);

            //直線の方程式求める
            float a, b;
            yagi::getGradSegment(pt1, pt2, &a, &b);
            imList[i].grads.push_back(a);
            imList[i].segments.push_back(b);

            //デバッグ用カラー画像にライン投影
            drawLine(dummy, pt1, pt2, 1, RED);
            drawLine(imList[i].trackLineAndOpenPoseImage, pt1, pt2, 1, RED);
            this->imList[i].trackLineImage = dummy.clone();

            gradAve += a;
            gradTxt << a << " ";
        }

        gradTxt << gradAve << endl;


        //そのフレームのトラック外をマスク
        imList[i].trackAreaMask = maskOutArea(imList[i].track_line_masks[imList[i].track_lines.size() - 1], imList[i].track_line_masks[0]);

//        cv::imshow("estimated lines", dummy);
//        int k = cv::waitKey(0);
//        if(k == 115) {
//            cout << imList[i].grads[0] << " " << imList[i].grads[7] << endl;
//            gradTxt << imList[i].grads[0] << " " << imList[i].grads[7] << endl;
//        }


        // 検出したトラックライン表示
        if (SHOW_TRACKLINES) {
            cv::imshow("estimated lines", dummy);
            cv::imwrite("../trackLine/image" + yagi::digitString(i, 4) + ".jpg", dummy);
            cv::waitKey(0);
        }
    }
    cout << "[Tracking track line finished]" << endl;
}

//void Panorama::trackTracking(bool debug) {
//
//    //カラー変数
//    cv::Scalar WHITE(255, 255, 255);
//    cv::Scalar RED(0, 0, 255);
//
//    int frame_counter = 0;
//
//    cout << "Track line detecting" << endl;
//    //全画像のループ
//
//    for (auto im = imList.begin(); (im + 1) != imList.end(); ++im) {
//
//        cv::Mat edge = im->edge;
//        cv::Mat dummy = im->image.clone();
//        dummy = im->image.clone();
//
//        //全レーンのループ
//        for (int line = 0; line < im->track_lines.size(); line++) {
//
//            cv::Mat line_neighbor = maskAofB(im->track_line_masks[line], edge);
//
//
//            //RANSAC + 最小二乗法で各直線求めマスク
//            cv::Point2f pt1, pt2;
//            calcLine(line_neighbor, &pt1, &pt2);
//
//
//            //求めた直線周辺をマスク
//            cv::Mat line_mask = cv::Mat::zeros(edge.size(), CV_8U);
//            drawLine(line_mask, pt1, pt2, 15, WHITE);
//
//
//            //直線の方程式求める
//            float a, b;
//            yagi::getGradSegment(pt1, pt2, &a, &b);
//
//
//            //デバッグ用カラー画像にライン投影
//            drawLine(dummy, pt1, pt2, 1, RED);
//            this->imList[frame_counter].trackLineImage = dummy.clone();
////            cv::imshow("estimated lines", dummy);
////            cv::waitKey(0);
//
//
//            //そのフレームでのトラック追跡ライン
//            pair<cv::Point2f, cv::Point2f> next_line(pt1, pt2);
//            (im + 1)->track_line_masks.push_back(line_mask);
//            (im + 1)->track_lines.push_back(next_line);
//            (im + 1)->grads.push_back(a);
//            (im + 1)->segments.push_back(b);
//        }
//
//        //そのフレームのトラック外をマスク
//        (im + 1)->trackAreaMask = maskOutArea(im->track_line_masks[0],
//                                              im->track_line_masks[im->track_lines.size() - 1]);
//
//        // 関節位置を投影
////        for (OpenPoseBody hb : im->Runners) {
////            int jointID = 0;
////            for (cv::Point2f pt : hb.bodyPts){
////                jointID++;
////                cv::circle(dummy, pt, 2, WHITE, 2);
////            }
////        }
//
//        // 検出したトラックライン表示
//        if (debug) {
//            cout << "frame " << frame_counter << endl;
//            cv::imshow("estimated lines", dummy);
//            cv::imwrite("../trackLine/image" + yagi::digitString(frame_counter, 4) + ".jpg", dummy);
//            cv::waitKey(0);
//        }
//        frame_counter++;
//    }
//    cout << "tracking track finished " << endl;
//}


cv::Mat maskOutArea(cv::Mat in, cv::Mat out) {
    cv::Mat track_mask = cv::Mat::zeros(in.rows, in.cols, CV_8U) * 255;

    for (int i = out.cols - 1; i >= 0; i--) {
        if (out.at<unsigned char>(0, i) == 255)
            break;
        for (int j = 0; j < out.rows; j++) {
            if (out.at<unsigned char>(j, i) == 255)
                break;
            track_mask.at<unsigned char>(j, i) = 255;
        }
    }

    for (int i = 0; i < in.cols; i++) {
        if (in.at<unsigned char>(in.rows - 1, i) == 255)
            break;
        for (int j = in.rows - 1; j >= 0; j--) {
            if (in.at<unsigned char>(j, i) == 255)
                break;
            track_mask.at<unsigned char>(j, i) = 255;
        }
    }

//    cv::imshow("aaa", track_mask);
//    cv::waitKey();

    return track_mask;
}



