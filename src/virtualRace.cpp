//
// Created by yagi on 18/10/02.
//

#include "virtualRace.h"
#include "basicFunctions/basicFunction.h"
using namespace std;
using namespace cv;

void virtualRace::readSavedData(std::string videoName){

    string textPath = "../projects/" + videoName + "/texts/";
    string homographyPath = textPath + "/homography.txt";
    string resizeHPath = textPath + "/resizeH.txt";
    string cornerPtPath = textPath + "/startFinishLine.txt";
    string panoramaSizePath = textPath + "/panoramaSize.txt";
    string str;
    vector<string> line_strings;

    //Load Homography matrix
    ifstream hTxt(homographyPath);
    while (getline(hTxt, str))
    {
        line_strings = yagi::split(str, ' ');
        cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                H.at<double>(i, j) = stod(line_strings[i * 3 + j]);
            }
        }
        this->HomographyList.push_back(H);
    }
    hTxt.close();

    //Load resize homography matrix
    ifstream reTxt(resizeHPath);
    while (getline(reTxt, str))
    {
        line_strings = yagi::split(str, ' ');
        cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                H.at<double>(i, j) = stod(line_strings[i * 3 + j + 2]);
            }
        }
        this->resizeH.push_back(H);
    }
    reTxt.close();

    //Load corner points
    ifstream ptTxt(cornerPtPath);
    while (getline(ptTxt, str))
    {
        line_strings = yagi::split(str, ' ');
        cv::Point2f pt(stof(line_strings[0]), stof(line_strings[1]));
        this->cornerPoints.push_back(pt);
    }
    ptTxt.close();

    //Load panoramaSize
    ifstream sizeTxt(panoramaSizePath);
    while (getline(sizeTxt, str))
    {
        line_strings = yagi::split(str, ' ');
        this->smallPanoramaWidth = stof(line_strings[0]);
        this->smallPanoramaHeight = stof(line_strings[1]);
    }
    sizeTxt.close();

    //Load images
    ifstream ifs( "../projects/" + videoName + "/texts/imagelist.txt");
    string line;
    while (getline(ifs, line)) {
        cv::Mat image = cv::imread(line);
        this->srcImages.push_back(image);
    }
}

void virtualRace::loadImages(string videoName) {
    ifstream ifs( "../projects/" + videoName + "/texts/imagelist.txt");
    string line;
    while (getline(ifs, line)) {
        cv::Mat image = cv::imread(line);
        this->srcImages.push_back(image);
    }
}