#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>


#include <filesystem> // or #include <filesystem> for C++17 and up

namespace fs = std::filesystem;

int main(int argc, char** argv) {

	if (argc < 4)
		return 0;

	std::string outp = argv[1];

	if (!fs::is_directory(outp) || !fs::exists(outp)) { // Check if src folder exists
		fs::create_directory(outp); // create src folder
	}



	std::string path = argv[2];
	std::string Templ = argv[3];

	for (const auto& entry : fs::directory_iterator(path))
	{
		std::cout << entry.path() << std::endl;

		std::size_t found = entry.path().u8string().find(".bmp");

		if (found == std::string::npos)
			continue;
		std::string img_file1(entry.path().u8string());
		std::string img_file2(Templ);

		cv::Mat img2 = cv::imread(img_file1, CV_LOAD_IMAGE_COLOR);
		cv::Mat img1 = cv::imread(img_file2, CV_LOAD_IMAGE_COLOR);

		cv::Mat Copp = img2.clone();

		//Canny(im1, im1, 180, 180 * 2, 3);
		//Canny(im0, im0, 180, 180 * 2, 3);


		if (img1.channels() != 1) {
			cvtColor(img1, img1, cv::COLOR_RGB2GRAY);
		}

		if (img2.channels() != 1) {
			cvtColor(img2, img2, cv::COLOR_RGB2GRAY);
		}

		img2 = 255 - img2;
		img1 = 255 - img1;

		threshold(img1, img1, 70, 255, cv::ThresholdTypes::THRESH_BINARY);
		threshold(img2, img2, 70, 255, cv::ThresholdTypes::THRESH_BINARY);

		//std::cout << img2.channels();

		 //cv::imwrite("dff.jpg", img2);
		 //cv::imshow("dffsd", img1);

		 //cv::waitKey(0);

		std::vector<cv::KeyPoint> kpts1;
		std::vector<cv::KeyPoint> kpts2;

		cv::Mat desc1;
		cv::Mat desc2;

		std::vector<cv::DMatch> matches;

		cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create();
		sift->detectAndCompute(img1, cv::Mat(), kpts1, desc1);
		sift->detectAndCompute(img2, cv::Mat(), kpts2, desc2);

		cv::Mat output1;
		cv::drawKeypoints(img1, kpts1, output1);
		//cv::imwrite("sift_result1.jpg", output1);

		cv::Mat output2;
		cv::drawKeypoints(img2, kpts2, output2);
		//cv::imwrite("sift_result2.jpg", output2);

		cv::BFMatcher desc_matcher(cv::NORM_L2, true);
		desc_matcher.match(desc1, desc2, matches, cv::Mat());

		std::vector<char> match_mask(matches.size(), 1);
		//findKeyPointsHomography(kpts1, kpts2, matches, match_mask);

		if (static_cast<int>(match_mask.size()) < 3) {
			return 0;
		}
		std::vector<cv::Point2f> pts1;
		std::vector<cv::Point2f> pts2;
		for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
			pts1.push_back(kpts1[matches[i].queryIdx].pt);
			pts2.push_back(kpts2[matches[i].trainIdx].pt);
		}
		findHomography(pts1, pts2, cv::RANSAC, 4, match_mask);

		cv::Mat stat, centroid;

		cv::Mat labelImage(img2.size(), CV_32S);
		int nLabels = cv::connectedComponentsWithStats(img2, labelImage, stat, centroid, 8);
		std::vector<uchar> colors(nLabels);

		std::vector<int> maps(nLabels, 0);

		for (int i = 0; i < pts2.size(); i++)
		{
			int cm = pts2[2].x;
			int rm = pts2[2].y;

			int label1 = labelImage.at<int>(rm, cm);

			maps[label1]++;
		}

		int labelmaybe(0), totcon = 0;

		for (int i = 1; i < maps.size(); i++)
		{
			if (!i)
			{
				labelmaybe = i;
				totcon = maps[i];
			}
			else if (totcon < maps[i])
			{
				labelmaybe = i;
				totcon = maps[i];
			}
		}

		std::cout << labelmaybe << std::endl;

		cv::Rect r(cv::Rect(cv::Point(stat.at<int>(labelmaybe, cv::CC_STAT_LEFT), stat.at<int>(labelmaybe, cv::CC_STAT_TOP)), cv::Size(stat.at<int>(labelmaybe, cv::CC_STAT_WIDTH), stat.at<int>(labelmaybe, cv::CC_STAT_HEIGHT))));

		cv::rectangle(Copp, r, cv::Scalar(0, 0, 255), 1);
		//std::cout << stat.at<int>(labelmaybe, cv::CC_STAT_AREA);
		/*if (0)
		{
			cv::imwrite("sds.jpg",labelImage);

			for (int i = 1; i < nLabels; i++)
			{
				std::cout << stat.at<int>(i, cv::CC_STAT_AREA) << std::endl;
				cv::Mat dst(img2.size(), CV_8UC1, cv::Scalar(0));
				for (int r = 0; r < dst.rows; ++r) {
					for (int c = 0; c < dst.cols; ++c) {
						int label = labelImage.at<int>(r, c);

						if (label == i)
						{
							uchar& pixel = dst.at<uchar>(r, c);
							pixel = 255;
						}
					}
				}

				cv::imwrite(std::to_string(i) + ".jpg", img2);
			}
		}*/

		cv::imwrite(outp + "//" +std::to_string(labelmaybe) + ".bmp", Copp);


		//imshow("Connected Components", Copp);

		//cv::waitKey(0);

		cv::Mat res;
		cv::drawMatches(img1, kpts1, img2, kpts2, matches, res, cv::Scalar::all(-1),
			cv::Scalar::all(-1), match_mask, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);



		cv::imwrite(outp +  "//matched" + std::to_string(labelmaybe) + ".bmp", res);

		//cv::imshow("result", res);
		//cv::waitKey(0);
	}
	return 0;
}