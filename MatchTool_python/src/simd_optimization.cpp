#include "simd_optimization.h"
#include <algorithm>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace MatchTool {

// SIMD detection
bool SIMDOptimizer::has_sse2() {
#if defined(__SSE2__)
    return true;
#else
    return false;
#endif
}

bool SIMDOptimizer::has_avx2() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

bool SIMDOptimizer::has_neon() {
#if defined(__ARM_NEON) || defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

// SSE2 implementation
#if defined(__SSE2__)
inline int32_t hsum_epi32_sse2(__m128i x) {
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

int32_t SIMDOptimizer::convolution_sse2(const uint8_t* kernel, const uint8_t* data, int length) {
    const int block_size = 16;
    const int block = length / block_size;
    __m128i sum_v = _mm_setzero_si128();

    for (int y = 0; y < block * block_size; y += block_size) {
        __m128i src_k = _mm_loadu_si128((__m128i*)(kernel + y));
        __m128i src_c = _mm_loadu_si128((__m128i*)(data + y));
        
        __m128i src_k_l = _mm_unpacklo_epi8(src_k, _mm_setzero_si128());
        __m128i src_k_h = _mm_unpackhi_epi8(src_k, _mm_setzero_si128());
        __m128i src_c_l = _mm_unpacklo_epi8(src_c, _mm_setzero_si128());
        __m128i src_c_h = _mm_unpackhi_epi8(src_c, _mm_setzero_si128());

        __m128i sum_l = _mm_madd_epi16(src_k_l, src_c_l);
        __m128i sum_h = _mm_madd_epi16(src_k_h, src_c_h);

        __m128i sum_t = _mm_add_epi32(sum_l, sum_h);
        sum_v = _mm_add_epi32(sum_v, sum_t);
    }

    int32_t sum = hsum_epi32_sse2(sum_v);

    for (int y = block * block_size; y < length; y++) {
        sum += kernel[y] * data[y];
    }

    return sum;
}
#else
int32_t SIMDOptimizer::convolution_sse2(const uint8_t* kernel, const uint8_t* data, int length) {
    int32_t sum = 0;
    for (int i = 0; i < length; ++i) {
        sum += kernel[i] * data[i];
    }
    return sum;
}
#endif

// AVX2 implementation
#if defined(__AVX2__)
inline int32_t hsum_epi32_avx2(__m256i x) {
    __m128i lo = _mm256_castsi256_si128(x);
    __m128i hi = _mm256_extracti128_si256(x, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

int32_t SIMDOptimizer::convolution_avx2(const uint8_t* kernel, const uint8_t* data, int length) {
    const int block_size = 32;
    const int block = length / block_size;
    __m256i sum_v = _mm256_setzero_si256();

    for (int y = 0; y < block * block_size; y += block_size) {
        __m256i src_k = _mm256_loadu_si256((__m256i*)(kernel + y));
        __m256i src_c = _mm256_loadu_si256((__m256i*)(data + y));
        
        __m256i src_k_l = _mm256_unpacklo_epi8(src_k, _mm256_setzero_si256());
        __m256i src_k_h = _mm256_unpackhi_epi8(src_k, _mm256_setzero_si256());
        __m256i src_c_l = _mm256_unpacklo_epi8(src_c, _mm256_setzero_si256());
        __m256i src_c_h = _mm256_unpackhi_epi8(src_c, _mm256_setzero_si256());

        __m256i sum_l = _mm256_madd_epi16(src_k_l, src_c_l);
        __m256i sum_h = _mm256_madd_epi16(src_k_h, src_c_h);

        __m256i sum_t = _mm256_add_epi32(sum_l, sum_h);
        sum_v = _mm256_add_epi32(sum_v, sum_t);
    }

    int32_t sum = hsum_epi32_avx2(sum_v);

    for (int y = block * block_size; y < length; y++) {
        sum += kernel[y] * data[y];
    }

    return sum;
}
#else
int32_t SIMDOptimizer::convolution_avx2(const uint8_t* kernel, const uint8_t* data, int length) {
    return convolution_sse2(kernel, data, length);
}
#endif

// ARM NEON implementation
#if defined(__ARM_NEON) || defined(__aarch64__)
inline int32_t vaddvq_s32(int32x4_t v) {
    int32x2_t tmp = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
    return vget_lane_s32(vpadd_s32(tmp, tmp), 0);
}

int32_t SIMDOptimizer::convolution_neon(const uint8_t* kernel, const uint8_t* data, int length) {
    const int block_size = 16;
    const int block = length / block_size;
    int32x4_t sum_v = vdupq_n_s32(0);
    uint8x16_t zero = vdupq_n_u8(0);

    for (int y = 0; y < block * block_size; y += block_size) {
        uint8x16_t src_k = vld1q_u8(kernel + y);
        uint8x16_t src_c = vld1q_u8(data + y);
        
        int16x8_t src_k_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_k)));
        int16x8_t src_k_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src_k)));
        int16x8_t src_c_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_c)));
        int16x8_t src_c_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src_c)));

        int32x4_t sum_l = vmull_s16(vget_low_s16(src_k_l), vget_low_s16(src_c_l));
        int32x4_t sum_h = vmull_s16(vget_high_s16(src_k_l), vget_high_s16(src_c_l));
        sum_v = vaddq_s32(sum_v, vaddq_s32(sum_l, sum_h));

        sum_l = vmull_s16(vget_low_s16(src_k_h), vget_low_s16(src_c_h));
        sum_h = vmull_s16(vget_high_s16(src_k_h), vget_high_s16(src_c_h));
        sum_v = vaddq_s32(sum_v, vaddq_s32(sum_l, sum_h));
    }

    int32_t sum = vaddvq_s32(sum_v);

    for (int y = block * block_size; y < length; y++) {
        sum += kernel[y] * data[y];
    }

    return sum;
}
#else
int32_t SIMDOptimizer::convolution_neon(const uint8_t* kernel, const uint8_t* data, int length) {
    return convolution_sse2(kernel, data, length);
}
#endif

// Generic convolution function
int32_t SIMDOptimizer::convolution(const uint8_t* kernel, const uint8_t* data, int length) {
    if (has_avx2()) {
        return convolution_avx2(kernel, data, length);
    } else if (has_sse2()) {
        return convolution_sse2(kernel, data, length);
    } else if (has_neon()) {
        return convolution_neon(kernel, data, length);
    } else {
        // Fallback to scalar implementation
        int32_t sum = 0;
        for (int i = 0; i < length; ++i) {
            sum += kernel[i] * data[i];
        }
        return sum;
    }
}

// Fast correlation using SIMD
void SIMDOptimizer::fast_correlation(const cv::Mat& src, const cv::Mat& templ, cv::Mat& result) {
    if (src.channels() != 1 || templ.channels() != 1) {
        cv::matchTemplate(src, templ, result, cv::TM_CCORR);
        return;
    }

    int result_cols = src.cols - templ.cols + 1;
    int result_rows = src.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);
    result.setTo(0.0f);

    const uint8_t* templ_ptr = templ.ptr<uint8_t>();
    int templ_length = templ.rows * templ.cols;

    // Use SIMD for faster computation
    for (int y = 0; y < result_rows; ++y) {
        const uint8_t* src_row_ptr = src.ptr<uint8_t>(y);
        float* result_row_ptr = result.ptr<float>(y);
        
        for (int x = 0; x < result_cols; ++x) {
            const uint8_t* src_ptr = src_row_ptr + x;
            result_row_ptr[x] = static_cast<float>(convolution(templ_ptr, src_ptr, templ_length));
        }
    }
}

// Normalized cross correlation
void SIMDOptimizer::normalized_cross_correlation(const cv::Mat& src, const cv::Mat& templ, cv::Mat& result) {
    // For simplicity, fall back to OpenCV's implementation for NCC
    // In a production version, you could implement SIMD-optimized NCC
    cv::matchTemplate(src, templ, result, cv::TM_CCOEFF_NORMED);
}

} // namespace MatchTool