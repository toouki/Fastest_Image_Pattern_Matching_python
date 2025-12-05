#pragma once

#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>

namespace MatchTool {

// SIMD-optimized convolution functions
class SIMDOptimizer {
public:
    // Check if SIMD is supported on current platform
    static bool has_sse2();
    static bool has_avx2();
    static bool has_neon();
    
    // Fast convolution using SIMD
    static int32_t convolution_sse2(const uint8_t* kernel, const uint8_t* data, int length);
    static int32_t convolution_avx2(const uint8_t* kernel, const uint8_t* data, int length);
    static int32_t convolution_neon(const uint8_t* kernel, const uint8_t* data, int length);
    
    // Generic convolution that chooses the best SIMD implementation
    static int32_t convolution(const uint8_t* kernel, const uint8_t* data, int length);
    
    // Matrix operations for template matching
    static void fast_correlation(const cv::Mat& src, const cv::Mat& templ, cv::Mat& result);
    static void normalized_cross_correlation(const cv::Mat& src, const cv::Mat& templ, cv::Mat& result);
};

// Template data structure for pyramid matching
struct TemplateData {
    std::vector<cv::Mat> pyramid;
    std::vector<cv::Scalar> templ_mean;
    std::vector<double> templ_norm;
    std::vector<double> inv_area;
    std::vector<bool> result_equal_1;
    bool is_pattern_learned;
    int border_color;
    
    TemplateData() : is_pattern_learned(false), border_color(0) {}
    
    void clear() {
        pyramid.clear();
        templ_mean.clear();
        templ_norm.clear();
        inv_area.clear();
        result_equal_1.clear();
        is_pattern_learned = false;
    }
    
    void resize(int size) {
        templ_mean.resize(size);
        templ_norm.resize(size, 0.0);
        inv_area.resize(size, 1.0);
        result_equal_1.resize(size, false);
    }
};

// Match result structure
struct MatchResult {
    cv::Point2d position;
    double score;
    double angle;
    cv::Rect roi;
    
    MatchResult() : position(0, 0), score(0.0), angle(0.0) {}
    MatchResult(cv::Point2d pos, double s, double a = 0.0) 
        : position(pos), score(s), angle(a) {}
};

// Match parameters
struct MatchParameters {
    int max_positions;
    double max_overlap;
    double score_threshold;
    double tolerance_angle;
    int min_reduce_area;
    bool use_simd;
    bool enable_subpixel;
    bool fast_mode;
    
    MatchParameters() 
        : max_positions(10)
        , max_overlap(0.1)
        , score_threshold(0.8)
        , tolerance_angle(5.0)
        , min_reduce_area(1000)
        , use_simd(true)
        , enable_subpixel(false)
        , fast_mode(false) {}
};

} // namespace MatchTool