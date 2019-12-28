//
// Created by yagi on 18/07/18.
//

#include "panorama.h"
#include <opencv2/xfeatures2d.hpp>
#include "basicFunctions/basicFunction.h"


using namespace std;
using namespace yagi;
using namespace cv;

//グローバル変数
struct mouseParam {
    int x;
    int y;
};
bool clicked_4corners = false;
cv::Point2f myclicked_point;


//コールバック関数
void myrunnerCallBackFunc(int eventType, int x, int y, int flags, void *userdata) {
    switch (eventType) {
        case cv::EVENT_LBUTTONUP:
            std::cout << x << " , " << y << std::endl;
            myclicked_point.x = x;
            myclicked_point.y = y;
            clicked_4corners = true;

    }
}


void Panorama::startFinishLineSelect() {

    cout << "[Click start line]" << endl;

    vector<cv::Scalar> colors;
    setColor(&colors);
    string file_name = _txt_folder + "/startFinishLine.txt";

    //最初のフレームでスタートラインクリック
    mouseParam mouseEvent;
    string windowName = "click start line (Q: finish clicking)";
    cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, myrunnerCallBackFunc, &mouseEvent);
    cv::Mat image = imList[0].image.clone();

    if (!checkFileExistence(file_name)) {
        while (1) {
            cv::imshow(windowName, image);
            int key = cv::waitKey(1);

            if (clicked_4corners) {
                //click point格納
                clicked_4corners = false;
                cv::circle(image, myclicked_point, 2, colors[0], 2);
                cv::Point2f pt(myclicked_point.x, myclicked_point.y);
                this->startLineCornerPoints.push_back(pt);
            }

            if (key == 'q')
                break;
        }

        //最後のフレームでゴールラインクリック
        cout << "Click finish line" << endl;

        windowName = "click finish line(Q: finish clicking, B: back to previous frame)";
        cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
        cv::setMouseCallback(windowName, myrunnerCallBackFunc, &mouseEvent);
        bool lineSelected = false;

        for (int i = imList.size() - 2; i > 0; i--) {
            cv::Mat lastImage = imList[i].image.clone();

            while (1) {
                cv::imshow(windowName, lastImage);
                int key = cv::waitKey(1);

                if (clicked_4corners) {
                    //click point格納
                    clicked_4corners = false;
                    cv::circle(lastImage, myclicked_point, 2, colors[0], 2);
                    cv::Point2f pt(myclicked_point.x, myclicked_point.y);
                    this->finishLineCornerPoints.push_back(pt);
                }

                if (key == 'b') {
                    break;
                }

                if (key == 'q') {
                    this->finalLineImageNum = i;
                    lineSelected = true;
                    break;
                }
            }
            if (lineSelected)
                break;
        }

        ofstream outputfile(file_name);
        for (cv::Point2f point: startLineCornerPoints){
            outputfile << point.x << " " << point.y;
            outputfile << endl;
        }
        for (cv::Point2f point: finishLineCornerPoints){
            outputfile << point.x << " " << point.y;
            outputfile << endl;
        }
        outputfile << this->finalLineImageNum;
        outputfile.close();

    }else{
        std::ifstream ifs(file_name);
        std::string str;
        vector<cv::Point2f> clicked_4cornersPoints;
        if (ifs.fail())
        {
            std::cerr << "クリックポイントが見つかりません" << std::endl;
        }

        int i = 0;
        while (getline(ifs, str))
        {
            if (i < 4) {
                vector<string> loaded = yagi::split(str, ' ');
                cv::Point2f loadPt(stof(loaded[0]), stof(loaded[1]));
                clicked_4cornersPoints.push_back(loadPt);
            }else{
                vector<string> loaded = yagi::split(str, ' ');
                this->finalLineImageNum = stoi(loaded[0]);
            }
            i++;
        }
        this->startLineCornerPoints.push_back(clicked_4cornersPoints[0]);
        this->startLineCornerPoints.push_back(clicked_4cornersPoints[1]);
        this->finishLineCornerPoints.push_back(clicked_4cornersPoints[2]);
        this->finishLineCornerPoints.push_back(clicked_4cornersPoints[3]);

    }

    float a, b, c, d;
    float lineNum = imList[0].grads.size();
    yagi::getGradSegment(startLineCornerPoints[0], startLineCornerPoints[1], &a, &b);
    yagi::getGradSegment(finishLineCornerPoints[0], finishLineCornerPoints[1], &c, &d);
    cv::Point pt1 = yagi::getCrossingPoint(a, b, imList[0].grads[0], imList[0].segments[0]);
    cv::Point pt2 = yagi::getCrossingPoint(a, b, imList[0].grads[lineNum - 1], imList[0].segments[lineNum - 1]);
    cv::Point pt3 = yagi::getCrossingPoint(c, d, imList[this->finalLineImageNum].grads[0], imList[this->finalLineImageNum].segments[0]);
    cv::Point pt4 = yagi::getCrossingPoint(c, d, imList[this->finalLineImageNum].grads[lineNum - 1], imList[this->finalLineImageNum].segments[lineNum - 1]);
    startLineCornerPoints[0] = pt1;
    startLineCornerPoints[1] = pt2;
    finishLineCornerPoints[0] = pt3;
    finishLineCornerPoints[1] = pt4;
//    cv::Mat dummyIm = imList[0].image;
//    cv::circle(dummyIm, pt1, 2, cv::Scalar(255,0,0), 2);
//    cv::circle(dummyIm, pt2, 2, cv::Scalar(255,0,0), 2);
//    cv::imshow("pt", dummyIm);
//    cv::waitKey();

    cv::destroyAllWindows();

    cout << "[Click start line finished]" << endl;
}
