#include "simd_optimization.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace py = pybind11;

namespace MatchToolPython {

using namespace MatchTool;

// Utility functions
int get_top_layer(const cv::Mat& templ, int min_dst_length) {
    int top_layer = 0;
    int min_reduce_area = min_dst_length * min_dst_length;
    int area = templ.cols * templ.rows;
    
    while (area > min_reduce_area) {
        area /= 4;
        top_layer++;
    }
    return top_layer;
}

// Learn pattern from template image
TemplateData learn_pattern_impl(const cv::Mat& templ, int min_reduce_area = 1000) {
    TemplateData data;
    
    if (templ.empty()) {
        throw std::runtime_error("Template image is empty");
    }
    
    int top_layer = get_top_layer(templ, min_reduce_area);
    
    // Build pyramid
    cv::buildPyramid(templ, data.pyramid, top_layer);
    
    data.border_color = cv::mean(templ).val[0] < 128 ? 255 : 0;
    data.resize(data.pyramid.size());
    
    // Precompute statistics for each pyramid level
    for (size_t i = 0; i < data.pyramid.size(); i++) {
        double inv_area = 1.0 / (data.pyramid[i].rows * data.pyramid[i].cols);
        cv::Scalar templ_mean, templ_sdv;
        double templ_norm = 0, templ_sum2 = 0;
        
        cv::meanStdDev(data.pyramid[i], templ_mean, templ_sdv);
        templ_norm = templ_sdv[0] * templ_sdv[0] + templ_sdv[1] * templ_sdv[1] + 
                    templ_sdv[2] * templ_sdv[2] + templ_sdv[3] * templ_sdv[3];
        
        if (templ_norm < DBL_EPSILON) {
            data.result_equal_1[i] = true;
        }
        
        templ_sum2 = templ_norm + templ_mean[0] * templ_mean[0] + templ_mean[1] * templ_mean[1] + 
                    templ_mean[2] * templ_mean[2] + templ_mean[3] * templ_mean[3];
        
        templ_sum2 /= inv_area;
        templ_norm = std::sqrt(templ_norm);
        templ_norm /= std::sqrt(inv_area);
        
        data.inv_area[i] = inv_area;
        data.templ_mean[i] = templ_mean;
        data.templ_norm[i] = templ_norm;
    }
    
    data.is_pattern_learned = true;
    return data;
}

// Match template using learned pattern
std::vector<MatchResult> match_template_impl(
    const cv::Mat& src, 
    const TemplateData& templ_data, 
    const MatchParameters& params) {
    
    std::vector<MatchResult> results;
    
    if (src.empty() || !templ_data.is_pattern_learned) {
        return results;
    }
    
    if (templ_data.pyramid.empty()) {
        return results;
    }
    
    // For simplicity, implement basic matching at the original resolution
    // In a full implementation, you would implement the multi-scale pyramid matching
    cv::Mat result;
    
    if (params.use_simd) {
        SIMDOptimizer::fast_correlation(src, templ_data.pyramid[0], result);
    } else {
        cv::matchTemplate(src, templ_data.pyramid[0], result, cv::TM_CCOEFF_NORMED);
    }
    
    // Find multiple matches
    cv::Mat result_mask = cv::Mat::ones(result.size(), CV_8UC1);
    
    for (int i = 0; i < params.max_positions; ++i) {
        double min_val, max_val;
        cv::Point min_loc, max_loc;
        
        cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc, result_mask);
        
        if (max_val < params.score_threshold) {
            break;
        }
        
        // Add match result
        results.emplace_back(
            cv::Point2d(max_loc.x + templ_data.pyramid[0].cols / 2.0, 
                       max_loc.y + templ_data.pyramid[0].rows / 2.0),
            max_val,
            0.0
        );
        
        // Mask out the found region to find other matches
        int mask_x = max_loc.x - templ_data.pyramid[0].cols / 2;
        int mask_y = max_loc.y - templ_data.pyramid[0].rows / 2;
        mask_x = std::max(0, mask_x);
        mask_y = std::max(0, mask_y);
        
        int mask_w = templ_data.pyramid[0].cols * (1 + params.max_overlap);
        int mask_h = templ_data.pyramid[0].rows * (1 + params.max_overlap);
        
        cv::Rect mask_rect(mask_x, mask_y, 
                          std::min(mask_w, result.cols - mask_x),
                          std::min(mask_h, result.rows - mask_y));
        
        if (mask_rect.x >= 0 && mask_rect.y >= 0 && 
            mask_rect.x + mask_rect.width <= result.cols && 
            mask_rect.y + mask_rect.height <= result.rows) {
            result_mask(mask_rect) = 0;
        }
    }
    
    return results;
}

// Conversion functions for Python
cv::Mat numpy_to_mat(py::array_t<uint8_t> input) {
    py::buffer_info buf = input.request();
    
    if (buf.ndim == 2) {
        // Grayscale image
        return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    } else if (buf.ndim == 3 && buf.shape[2] == 3) {
        // RGB image - convert to BGR for OpenCV
        cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
        cv::Mat bgr_img;
        cv::cvtColor(img, bgr_img, cv::COLOR_RGB2BGR);
        return bgr_img;
    } else {
        throw std::runtime_error("Unsupported image format. Expected 2D (grayscale) or 3D (RGB) array.");
    }
}

py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array_t<uint8_t>();
    }
    
    if (mat.channels() == 1) {
        // Grayscale
        return py::array_t<uint8_t>(
            {mat.rows, mat.cols},
            {sizeof(uint8_t) * mat.cols, sizeof(uint8_t)},
            mat.data
        );
    } else if (mat.channels() == 3) {
        // Convert BGR to RGB
        cv::Mat rgb_img;
        cv::cvtColor(mat, rgb_img, cv::COLOR_BGR2RGB);
        return py::array_t<uint8_t>(
            {rgb_img.rows, rgb_img.cols, 3},
            {sizeof(uint8_t) * rgb_img.cols * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)},
            rgb_img.data
        );
    } else {
        throw std::runtime_error("Unsupported number of channels for conversion to numpy array");
    }
}

// Python wrapper classes
class PyTemplateData {
private:
    TemplateData data_;
    
public:
    PyTemplateData(const py::array_t<uint8_t>& templ, int min_reduce_area = 1000) {
        data_ = learn_pattern_impl(numpy_to_mat(templ), min_reduce_area);
    }
    
    bool is_learned() const { return data_.is_pattern_learned; }
    int pyramid_levels() const { return data_.pyramid.size(); }
    
    // For internal use by match_template
    const TemplateData& get_data() const { return data_; }
};

// Template matching class
class PyTemplateMatcher {
private:
    TemplateData templ_data_;
    MatchParameters params_;
    
public:
    PyTemplateMatcher() = default;
    
    void learn_pattern(const py::array_t<uint8_t>& templ, int min_reduce_area = 1000) {
        templ_data_ = learn_pattern_impl(numpy_to_mat(templ), min_reduce_area);
    }
    
    void set_parameters(const MatchParameters& params) {
        params_ = params;
    }
    
    py::list match(const py::array_t<uint8_t>& src) {
        auto results = match_template_impl(numpy_to_mat(src), templ_data_, params_);
        
        py::list py_results;
        for (const auto& result : results) {
            py::dict py_result;
            py_result["position"] = py::make_tuple(result.position.x, result.position.y);
            py_result["score"] = result.score;
            py_result["angle"] = result.angle;
            py_results.append(py_result);
        }
        return py_results;
    }
    
    py::array_t<uint8_t> visualize_matches(const py::array_t<uint8_t>& src) {
        cv::Mat src_mat = numpy_to_mat(src);
        auto results = match_template_impl(src_mat, templ_data_, params_);
        
        cv::Mat vis;
        if (src_mat.channels() == 1) {
            cv::cvtColor(src_mat, vis, cv::COLOR_GRAY2BGR);
        } else {
            vis = src_mat.clone();
        }
        
        // Draw rectangles around matches
        for (const auto& result : results) {
            cv::Point tl(result.position.x - templ_data_.pyramid[0].cols / 2,
                        result.position.y - templ_data_.pyramid[0].rows / 2);
            cv::Point br(result.position.x + templ_data_.pyramid[0].cols / 2,
                        result.position.y + templ_data_.pyramid[0].rows / 2);
            cv::rectangle(vis, tl, br, cv::Scalar(0, 255, 0), 2);
            
            // Add score text
            std::string score_text = std::to_string(result.score);
            cv::putText(vis, score_text, tl, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(0, 255, 0), 1);
        }
        
        return mat_to_numpy(vis);
    }
};

} // namespace MatchToolPython

// Module definition
PYBIND11_MODULE(core_matcher, m) {
    m.doc() = "Fast Image Pattern Matching with SIMD Optimization";
    
    using namespace MatchToolPython;
    using namespace MatchTool;
    
    // Struct bindings
    py::class_<MatchParameters>(m, "MatchParameters")
        .def(py::init<>())
        .def_readwrite("max_positions", &MatchParameters::max_positions)
        .def_readwrite("max_overlap", &MatchParameters::max_overlap)
        .def_readwrite("score_threshold", &MatchParameters::score_threshold)
        .def_readwrite("tolerance_angle", &MatchParameters::tolerance_angle)
        .def_readwrite("min_reduce_area", &MatchParameters::min_reduce_area)
        .def_readwrite("use_simd", &MatchParameters::use_simd)
        .def_readwrite("enable_subpixel", &MatchParameters::enable_subpixel)
        .def_readwrite("fast_mode", &MatchParameters::fast_mode);
    
    py::class_<MatchResult>(m, "MatchResult")
        .def(py::init<>())
        .def_readonly("position", &MatchResult::position)
        .def_readonly("score", &MatchResult::score)
        .def_readonly("angle", &MatchResult::angle)
        .def_readonly("roi", &MatchResult::roi);
    
    // Main classes
    py::class_<PyTemplateData>(m, "PyTemplateData")
        .def(py::init<const py::array_t<uint8_t>&, int>(), py::arg("templ"), py::arg("min_reduce_area") = 1000)
        .def("is_learned", &PyTemplateData::is_learned)
        .def("pyramid_levels", &PyTemplateData::pyramid_levels);
    
    py::class_<PyTemplateMatcher>(m, "TemplateMatcher")
        .def(py::init<>())
        .def("learn_pattern", &PyTemplateMatcher::learn_pattern, 
             py::arg("templ"), py::arg("min_reduce_area") = 1000)
        .def("set_parameters", &PyTemplateMatcher::set_parameters)
        .def("match", &PyTemplateMatcher::match)
        .def("visualize_matches", &PyTemplateMatcher::visualize_matches);
    
    // Utility functions
    m.def("get_version", []() { return "0.1.0"; });
    m.def("has_simd_support", []() { 
        return SIMDOptimizer::has_sse2() || SIMDOptimizer::has_avx2() || SIMDOptimizer::has_neon();
    });
    
    // Convenience functions
    m.def("learn_pattern", [](const py::array_t<uint8_t>& templ, int min_reduce_area) {
        return PyTemplateData(templ, min_reduce_area);
    }, py::arg("templ"), py::arg("min_reduce_area") = 1000);
    
    m.def("match_template", [](const py::array_t<uint8_t>& src, 
                               const PyTemplateData& templ_data,
                               const MatchParameters& params = MatchParameters()) {
        auto matcher = std::make_shared<PyTemplateMatcher>();
        // This is a simplified version - in practice you'd need to extract the internal data
        return py::list(); // Placeholder
    });
}