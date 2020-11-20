#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/types.hpp>

#include <stdio.h>
#include <iostream>

cv::RNG rng(12345);

int main(int argc, char** argv)
{
    cv::VideoCapture video;
    video.open("./../videos/rgbd_dataset_freiburg2_pioneer_slam3-rgb.avi");

    cv::Mat frame, oldFrame, mask, greyscaleFrame, desc, homographyMask;
    std::vector<cv::KeyPoint> kps;
    std::vector<std::vector<cv::KeyPoint>> kpsList;
    std::vector<cv::Mat> descList;

    cv::SIFT surface;
    surface.create();

    for (;;)
    {
        video.read(frame);
        if (frame.empty())
        {
            std::cout << "empty frame" << std::endl;
            break;
        }

        cv::cvtColor(frame, greyscaleFrame, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> features;
        cv::goodFeaturesToTrack(greyscaleFrame, features, 3000, 0.02f, 7.0f);

        std::vector<cv::KeyPoint> keypoints;
        for (size_t i = 0; i < features.size(); i++)
        {
            keypoints.push_back(cv::KeyPoint(features[i], 1.f));
        }

        kpsList.push_back(keypoints);

        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->compute(greyscaleFrame, keypoints, desc);
        descList.push_back(desc);

        if (descList.size() > 1)
        {
            std::vector<std::vector<cv::DMatch>> knnMatches;
            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
            matcher->knnMatch(descList[descList.size() - 1], descList[descList.size() - 2], knnMatches, 2);

            const float ratio_thresh = 0.75f;
            std::vector<cv::DMatch> good_matches;
            for (size_t i = 0; i < knnMatches.size(); i++)
            {
                if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance)
                {
                    good_matches.push_back(knnMatches[i][0]);
                }
            }

            std::vector<cv::Point2f> obj;
            std::vector<cv::Point2f> scene;
            for (size_t i = 0; i < good_matches.size(); i++)
            {
                //-- Get the keypoints from the good matches
                obj.push_back(kpsList[kpsList.size() - 1][good_matches[i].queryIdx].pt);
                scene.push_back(kpsList[kpsList.size() - 2][good_matches[i].trainIdx].pt);
            }

            std::cout << "Test" << std::endl;

            cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC, 3, homographyMask, 3000);
            // cv::Mat H = cv::findEssentialMat(obj, scene, cv::RANSAC, 3, homographyMask, 3000);

            int maskRow = 10;
            std::cout << "POINTS: object(" << obj.at(maskRow).x << "," << obj.at(maskRow).y << ") - scene(" << scene.at(maskRow).x << "," << scene.at(maskRow).y << ")" << std::endl;
            std::cout << "mask value for 10'th row" << (unsigned int)homographyMask.at<uchar>(maskRow) << std::endl;
            std::cout << "mask value for 20'th row" << (unsigned int)homographyMask.at<uchar>(20) << std::endl;

            std::vector<cv::DMatch> better_matches;
            for (size_t i = 0; i < good_matches.size(); i++)
            {
                if ((unsigned int)homographyMask.at<uchar>(i) > 0)
                {
                    better_matches.push_back(good_matches[i]);
                }
            }

            std::cout << "Test" << std::endl;
            // std::cout << "H rows, cols: " << H.rows << H.cols << std::endl;
            std::cout << "H mask rows: " << homographyMask.rows << " cols: : " << homographyMask.cols << std::endl;
            std::cout << "Good matches size:" << good_matches.size() << std::endl;
            std::cout << "Better matches size:" << better_matches.size() << std::endl;
            std::cout << "Good - better diff size:" << good_matches.size() - better_matches.size() << std::endl;

            // std::cout << "H mask: " << homographyMask << std::endl;

            // std::cout << "H size:" << H.size() << std::endl;
            // std::cout << "H: " << H << std::endl;

            cv::Mat img_matches;

            // for (int i = 0; i < (int)better_matches.size(); i++)
            // {
            //     cv::Point2f point_old = kpsList[kpsList.size() - 2][better_matches[i].queryIdx].pt;
            //     cv::Point2f point_new = kpsList[kpsList.size() - 1][better_matches[i].trainIdx].pt;
            //     cv::circle(frame, point_old, 3, cv::Scalar(0, 0, 255), 1);
            //     cv::circle(frame, point_new, 3, cv::Scalar(255, 0, 0), 1);
            //     cv::line(frame, point_old, point_new, cv::Scalar(0, 255, 0), 2, 8, 0);
            // }

            std::cout << "Test" << std::endl;
            cv::drawMatches(frame, kpsList[kpsList.size() - 1], oldFrame, kpsList[kpsList.size() - 2], good_matches, img_matches, cv::Scalar::all(-1),
                            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imshow("Video", img_matches);

            // int i;
            // std::cin >> i;
        }

        // int radius = 4;
        // for (size_t i = 0; i < features.size(); i++)
        // {
        //     cv::circle(frame, features[i], radius, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 256), rng.uniform(0, 256)), cv::FILLED);
        // }

        //

        oldFrame = frame;

        if (cv::waitKey(5) >= 0)
            break;
    }

    return 0;
}