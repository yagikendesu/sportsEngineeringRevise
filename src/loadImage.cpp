//
// Created by yagi on 18/01/10.
//

#include "panorama.h"
#include "basicFunctions/basicFunction.h"

using namespace yagi;
using namespace std;

void Panorama::loadImage() {

    //画像リストopen
    ifstream ifs(_image_list_path);

    //imshowうまくいかない時用
    string line;
    int string_size = 0;

    //画像枚数カウンター
    int line_counter = 0;

    cv::Mat preEdge = cv::Mat::zeros(1,1, CV_8U);
    // cv::namedWindow("image", CV_WINDOW_NORMAL);
    while (getline(ifs, line)) {

//        //imshowがうまくいかないときここ原因(下4行をコメントアウト)
//        if(this->_video_name == "kiryu998" || this->_video_name == "2016_Rio" || this->_video_name == "100mT42") {
//            if (string_size == 0 || (string_size + 1) == line.size()) {
//                line.erase(line.size() - 1);
//            }
//            string_size = line.size();
//        }

        //ImageInfoに画像を格納していく
        Panorama::ImageInfo image_info;

        //img_namesに画像の名前格納
        this->img_names.push_back(line);

        //カラー、グレースケール,hsv
        cv::Mat image = cv::imread(line);
        cv::Mat gray_image = cv::imread(line, 0);
        cv::Mat img_hsv;
        cv::cvtColor(image, img_hsv, cv::COLOR_BGR2HSV);

        //エッジ画像（縦方向微分）
        cv::Mat edge = cv::Mat::zeros(gray_image.size(), CV_8UC1);
        for (int i = 1; i < gray_image.rows; i ++){
            for (int j = 0; j < gray_image.cols; j ++){
                if ((gray_image.at<unsigned char>(i, j) - gray_image.at<unsigned char>(i - 1, j)) > 30){
                    edge.at<unsigned char>(i, j) = 255;
                }
            }
        }

        //エッジ画像（横方向微分）
        /// Generate grad_x and grad_y
        int scale = 1;
        int delta = 0;
        int ddepth = CV_8U;
        cv::Mat grad_x;
        cv::Mat abs_grad_x, abs_grad_y;
        cv::Sobel(gray_image, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( grad_x, abs_grad_x );


        image_info.image = image;
        image_info.gray_image = gray_image;
        image_info.hsv_image = img_hsv;
        image_info.edge = edge;
        image_info.edge_horizontal = grad_x;
        image_info.trackLineAndOpenPoseImage = cv::Mat::zeros(image.size(), CV_8UC3);
        image_info.frameID = line_counter;
        preEdge = gray_image;

        //DensePoseのマスク
//        cout << this->_image_folder + _video_name + "/maskImages/image" + yagi::digitString(line_counter, 4) + "_IUV.png" << endl;
        cv::Mat mask = cv::imread(this->_image_folder + _video_name + "/maskImages/image" + yagi::digitString(line_counter, 4) + "_IUV.png", 0);
        cv::threshold(mask, mask, 10, 255, cv::THRESH_BINARY);
        image_info.denseMask = mask;

        imList.push_back(image_info);

        //デバック
        if (SHOW_LOADED_IMAGE) {
            cv::imshow("color", edge);
//            cv::imshow("gray", gray_image);
            cout << "Frame: " << line_counter << endl;
            cv::waitKey(0);
        }
        line_counter++;
    }
    IMG_WIDTH = imList[0].image.cols;
    IMG_HEIGHT = imList[0].image.rows;
}
