/// C++
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

/// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

// ChartDirector
// #include <chartdir.h>

/// Additional
#include "CourtModel.h"

//#define DLLEXPORT __declspec(dllexport)

int** CourtModel(std::string filename) {
    /***
    :params filename: a video, mp4-file or vmv-file
    :return cm: matrix 12 * 4,
    ***/
	// open a video file
//	std::string filename = "videos/1/1.wmv";
	struct stat buffer;
	if (stat(filename.c_str(), &buffer) != 0) {
		std::cerr << "Unable to open video file: " << filename << " !" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Load " << filename << std::endl;

	// court detection
	std::vector<cv::Vec4i> courtmodel;
	std::cout << "Court detection is in progress..." << std::endl;
	courtmodel = courtmodel::CreateCourtModel(filename);
	std::cout << "Court detection is finished!" << std::endl << std::endl;

	int** cm = (int**)malloc(12 * sizeof(int*));
	for(int i=0;i<12;i++) {
	    cm[i] = (int*)malloc(4 * sizeof(int));
	    cm[i][0] = courtmodel[i][0];
	    cm[i][1] = courtmodel[i][1];
	    cm[i][2] = courtmodel[i][2];
	    cm[i][3] = courtmodel[i][3];
	}
	return cm;
}

int main() {
    std::string filename = "ff_e_04.mp4";
    int** cm = CourtModel(filename);
    std::cout << "----------- cm start --------" << std::endl;
    for(int i=0;i<12;i++) {
        for(int j=0;j<4;j++)
            std::cout << cm[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << "----------- cm end --------" << std::endl;

    cv::Mat img = cv::imread("court4.png");
    for(int i=0;i<12;i++) {
        cv::Point p1(cm[i][0], cm[i][1]);
        cv::Point p2(cm[i][2], cm[i][3]);
        cv::line(img, p1, p2, cv::Scalar(0, 0, 0), 2);
    }
    cv::imshow("court", img);
    cv::waitKey(100000);
    return 0;
}
