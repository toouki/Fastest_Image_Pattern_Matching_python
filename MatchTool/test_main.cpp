#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

// Forward declare the SIMD function from x86_ubuntu_linux.cpp
extern int32_t IM_Conv_SIMD(uint8_t* pCharKernel, uint8_t* pCharConv, int iLength);

int main() {
    std::cout << "Fastest Image Pattern Matching - Linux Test" << std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    
    // Test basic OpenCV functionality
    cv::Mat testImage = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::rectangle(testImage, cv::Point(10, 10), cv::Point(50, 50), cv::Scalar(0, 255, 0), -1);
    
    std::cout << "Test image created: " << testImage.rows << "x" << testImage.cols << std::endl;
    
    // Test SIMD function with small data
    uint8_t kernel[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint8_t conv[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    int32_t result = IM_Conv_SIMD(kernel, conv, 16);
    std::cout << "SIMD test result: " << result << " (expected: 136)" << std::endl;
    
    std::cout << "Linux test completed successfully!" << std::endl;
    return 0;
}