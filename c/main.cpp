#include <opencv2/core.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/core/types.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <thread>
#include <chrono>

cv::RNG rng(12345);

int main(int argc, char **argv)
{
    cv::VideoCapture video;
    video.open("./../../videos/test_countryroad.mp4");

    cv::Mat frame, dispFrame, oldFrame, mask, greyscaleFrame, homographyMask;
    cv::cuda::GpuMat imageGpu, desc_gpu;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    cv::cuda::SURF_CUDA surf;
    // cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(500, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20, true);
    std::vector<cv::cuda::GpuMat> kps;
    std::vector<std::vector<cv::KeyPoint>> kpsList;
    std::vector<cv::Mat> descList;

    std::vector<cv::cuda::GpuMat> descList_gpu;
    std::vector<cv::cuda::GpuMat> kpsList_gpu;

    for (;;)
    {
        video.read(frame);
        if (frame.empty())
        {
            std::cout << "empty frame" << std::endl;
            break;
        }
        dispFrame = frame.clone();

        cv::cvtColor(frame, greyscaleFrame, cv::COLOR_BGR2GRAY);
        imageGpu.upload(greyscaleFrame);

        std::vector<cv::KeyPoint> keypoints;
        std::vector<float> desc;
        cv::cuda::GpuMat keypoints_gpu;
        surf.detectWithDescriptors(imageGpu, cv::cuda::GpuMat(), keypoints_gpu, desc_gpu);
        // surf.downloadKeypoints(keypoints_gpu, keypoints);
        // kpsList.push_back(keypoints);
        surf.downloadDescriptors(desc_gpu, desc);
        cv::Mat desc_tmp(desc);
        descList.push_back(desc_tmp);

        descList_gpu.push_back(desc_gpu);

        if (descList.size() > 1)
        {
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(descList[descList.size() - 1], descList_gpu[descList_gpu.size() - 2], knnMatches, 2);

            std::cout << "line 70" << std::endl;

            const float ratio_thresh = 0.75f;
            std::vector<cv::DMatch> good_matches;
            for (size_t i = 0; i < knnMatches.size(); i++)
            {
                if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance)
                {
                    good_matches.push_back(knnMatches[i][0]);
                }
            }

            surf.downloadKeypoints(keypoints_gpu, keypoints);
            // surf.downloadDescriptors(desc_gpu, desc);
            kpsList.push_back(keypoints);

            std::vector<cv::Point2f> obj;
            std::vector<cv::Point2f> scene;
            for (size_t i = 0; i < good_matches.size(); i++)
            {

                //-- Get the keypoints from the good matches
                obj.push_back(kpsList[kpsList.size() - 1][good_matches[i].queryIdx].pt);
                scene.push_back(kpsList[kpsList.size() - 2][good_matches[i].trainIdx].pt);
            }

            std::cout << "Test" << std::endl;

            // cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC, 5, homographyMask);

            // int maskRow = 10;
            // std::cout << "POINTS: object(" << obj.at(maskRow).x << "," << obj.at(maskRow).y << ") - scene(" << scene.at(maskRow).x << "," << scene.at(maskRow).y << ")" << std::endl;
            // std::cout << "mask value for 10'th row" << (unsigned int)homographyMask.at<int>(maskRow) << std::endl;
            // std::cout << "mask value for 20'th row" << (unsigned int)homographyMask.at<int>(20) << std::endl;

            // std::vector<cv::DMatch> better_matches;
            // for (size_t i = 0; i < homographyMask.rows; i++)
            // {
            //     if ((int)homographyMask.at<int>(i, 0) == 1)
            //     {

            //         better_matches.push_back(good_matches[i]);
            //     }
            // }

            std::cout << "Test" << std::endl;
            // std::cout << "H rows, cols: " << H.rows << H.cols << std::endl;
            // std::cout << "H mask rows: " << homographyMask.rows << " cols: : " << homographyMask.cols << std::endl;
            std::cout << "Good matches size:" << good_matches.size() << std::endl;
            // std::cout << "Better matches size:" << better_matches.size() << std::endl;
            // std::cout << "Good - better diff size:" << good_matches.size() - better_matches.size() << std::endl;

            // std::cout << "H mask: " << homographyMask << std::endl;

            // std::cout << "H size:" << H.size() << std::endl;
            // std::cout << "H: " << H << std::endl;

            cv::Mat img_matches;

            for (int i = 0; i < (int)good_matches.size(); i++)
            {
                cv::Point2f point_old = kpsList[kpsList.size() - 2][good_matches[i].queryIdx].pt;
                cv::Point2f point_new = kpsList[kpsList.size() - 1][good_matches[i].trainIdx].pt;
                cv::circle(dispFrame, point_old, 3, cv::Scalar(0, 0, 255), 1);
                cv::circle(dispFrame, point_new, 3, cv::Scalar(255, 0, 0), 1);

                cv::line(dispFrame, point_old, point_new, cv::Scalar(0, 255, 0), 2, 8, 0);
            }

            std::cout << "Test" << std::endl;
            // cv::drawMatches(frame, kpsList[kpsList.size() - 1], oldFrame, kpsList[kpsList.size() - 2], better_matches, img_matches, cv::Scalar::all(-1),
            //                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imshow("Video", dispFrame);
            // cv::imshow("Video", img_matches);

            // std::this_thread::sleep_for (std::chrono::milliseconds(1000));
            // int i;
            // std::cin >> i;
        }

        // int radius = 4;
        // for (size_t i = 0; i < features.size(); i++)
        // {
        //     cv::circle(frame, features[i], radius, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), cv::FILLED);
        // }

        //

        oldFrame = frame.clone();

        if (cv::waitKey(5) >= 0)
            break;
    }

    return 0;
}