//
// Created by yagi on 18/10/17.
//
// ----------------------------- OpenPose C++ API Tutorial - Example 1 - Body from image -----------------------------
// It reads an image, process it, and displays it with the pose keypoints.

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_POSE

#include "myOpenPose.h"
#include "../../src/basicFunctions/basicFunction.h"
#include<iostream>
#include<fstream>

using namespace cv;
using namespace std;


// Custom OpenPose flags
// Producer
//DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg", "Process an image. Read all standard formats (jpg, png, bmp, etc.).");

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr)
{
    // User's displaying/saving/other processing here
    // datum.cvOutputData: rendered frame with pose or heatmaps
    // datum.poseKeypoints: Array<float> with the estimated pose
    if (datumsPtr != nullptr && !datumsPtr->empty())
    {
        // Display image
        cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
        cv::waitKey(1);
    }
    else
        op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
}

void printKeypoints(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr)
{
    // Example: How to use the pose keypoints
    if (datumsPtr != nullptr && !datumsPtr->empty())
    {
        // Alternative 1
//        op::log("Body keypoints: " + datumsPtr->at(0).poseKeypoints.toString());

        // // Alternative 2
//        op::log(datumsPtr->at(0).poseKeypoints);

        // // Alternative 3
        cout << datumsPtr->at(0).poseKeypoints.getVolume() << endl;
        std::cout << datumsPtr->at(0).poseKeypoints << std::endl;

        // // Alternative 4 - Accesing each element of the keypoints
        // op::log("\nKeypoints:");
        // const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
        // op::log("Person pose keypoints:");
        // for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
        // {
        //     op::log("Person " + std::to_string(person) + " (x, y, score):");
        //     for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
        //     {
        //         std::string valueToPrint;
        //         for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
        //             valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
        //         op::log(valueToPrint);
        //     }
        // }
        // op::log(" ");
    }
    else
        op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
}

void yagi::outputTextFromVideo(const std::string video_path, const std::string image_output_path, const std::string txt_output_path){

    Mat img;
    VideoCapture cap(video_path); //Windowsの場合　パス中の¥は重ねて¥¥とする

    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};

    opWrapper.start();

    ofstream outputfile(txt_output_path + "human_pose_info.txt");
    ofstream imageoutputfile(txt_output_path + "op_imagelist.txt");

    //ディレクトリ作成
    const char *cstr = image_output_path.c_str();
    if (mkdir(cstr, 0777) == 0) {
        printf("directory correctly generated\n");
    } else {
        printf("directory already exists\n");
    }

    //エッジ強調
    const float k = -1.0;
    Mat sharpningKernel4 = Mat::zeros(3,3,CV_32F);
    sharpningKernel4.at<float>(0,1) = k;
    sharpningKernel4.at<float>(1,0) = k;
    sharpningKernel4.at<float>(1,1) = 6.0;
    sharpningKernel4.at<float>(1,2) = k;
    sharpningKernel4.at<float>(2,1) = k;

//    Mat sharpningKernel8 = (Mat_<float>(3, 3) << k, k, k, k, 9.0, k, k, k, k);

    cv::Rect rect(200, 200, 600, 400);

    // Process and display image
    bool personFound = false;
    int max_frame=cap.get(CV_CAP_PROP_FRAME_COUNT); //フレーム数
    for(int i=0; i<max_frame ;i++){ ; //1フレーム分取り出してimgに保持させる
        cap>>img ; //1フレーム分取り出してimgに保持させる
        if(img.cols == 0)
            break;
            //cv::Mat crop(img, rect);
            //img = crop;
             // 先鋭化フィルタを適用する
           // cv::filter2D(img, img, img.depth(), sharpningKernel4);
            cv::resize(img, img, Size(), 640.0 / img.cols, 320.0 / img.rows);
            //cv::Mat dummy = cv::Mat::zeros(img.cols, img.rows * (img.), CV_32FC3);

            auto datumProcessed = opWrapper.emplaceAndPop(img);
            if (datumProcessed != nullptr) {
//            printKeypoints(datumProcessed);
                if (datumProcessed != nullptr && !datumProcessed->empty()) {
                    outputfile << "Frame: " << i << endl;
                    personFound = false;
                    int elem_num = datumProcessed->at(0).poseKeypoints.getVolume();
                    for (int j = 0; j < elem_num; j++) {
                        if (j % 75 == 0) {
                            outputfile << "Person " << (j / 75) << " (x, y, score):" << endl;
                        }
                        outputfile << datumProcessed->at(0).poseKeypoints[j] << " ";
                        if (j % 75 == 74) {
                            outputfile << endl << "end";
                        }

                        if (j % 3 == 2) {
                            outputfile << endl;
                        }
                    }

                } else
                    op::log("Nullptr or empty datumsPtr found.", op::Priority::High);

                display(datumProcessed);
                cv::imwrite(image_output_path + yagi::digitString(i, 4) + ".jpg", datumProcessed->at(0).cvOutputData);
                imageoutputfile << image_output_path + yagi::digitString(i, 4) + ".jpg" << endl;
            } else
                op::log("Image could not be processed.", op::Priority::High);

    }

    outputfile.close();
    // Return successful message
    op::log("Stopping OpenPose...", op::Priority::High);
}

