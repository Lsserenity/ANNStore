#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif

#include <hnswlib/hnswlib.h>

namespace hnswlib {

// Squared L2 distance between two uint8 vectors, accumulated in uint32.
static float
L2SqrUint8(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);
    size_t qty = *static_cast<const size_t *>(qty_ptr);

    uint32_t acc = 0;

#if defined(__AVX2__)
    const size_t kStep = 32;
    size_t i = 0;
    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();

    for (; i + kStep <= qty; i += kStep) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVect1 + i));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pVect2 + i));

        __m128i a_lo = _mm256_castsi256_si128(a);
        __m128i a_hi = _mm256_extracti128_si256(a, 1);
        __m128i b_lo = _mm256_castsi256_si128(b);
        __m128i b_hi = _mm256_extracti128_si256(b, 1);

        __m256i a16_lo = _mm256_cvtepu8_epi16(a_lo);
        __m256i a16_hi = _mm256_cvtepu8_epi16(a_hi);
        __m256i b16_lo = _mm256_cvtepu8_epi16(b_lo);
        __m256i b16_hi = _mm256_cvtepu8_epi16(b_hi);

        __m256i d16_lo = _mm256_sub_epi16(a16_lo, b16_lo);
        __m256i d16_hi = _mm256_sub_epi16(a16_hi, b16_hi);

        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(d16_lo, d16_lo));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(d16_hi, d16_hi));
    }

    __m256i sum = _mm256_add_epi32(sum0, sum1);
    __m128i sum_hi = _mm256_extracti128_si256(sum, 1);
    __m128i sum_lo = _mm256_castsi256_si128(sum);
    __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);

    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
    acc += static_cast<uint32_t>(_mm_cvtsi128_si32(sum128));

    for (; i < qty; ++i) {
        const int diff = static_cast<int>(pVect1[i]) - static_cast<int>(pVect2[i]);
        acc += static_cast<uint32_t>(diff * diff);
    }

#elif defined(__SSE2__)
    const size_t kStep = 16;
    size_t i = 0;
    __m128i sum = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();

    for (; i + kStep <= qty; i += kStep) {
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVect1 + i));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pVect2 + i));

        __m128i a_lo = _mm_unpacklo_epi8(a, zero);
        __m128i a_hi = _mm_unpackhi_epi8(a, zero);
        __m128i b_lo = _mm_unpacklo_epi8(b, zero);
        __m128i b_hi = _mm_unpackhi_epi8(b, zero);

        __m128i d_lo = _mm_sub_epi16(a_lo, b_lo);
        __m128i d_hi = _mm_sub_epi16(a_hi, b_hi);

        sum = _mm_add_epi32(sum, _mm_madd_epi16(d_lo, d_lo));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(d_hi, d_hi));
    }

    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    acc += static_cast<uint32_t>(_mm_cvtsi128_si32(sum));

    for (; i < qty; ++i) {
        const int diff = static_cast<int>(pVect1[i]) - static_cast<int>(pVect2[i]);
        acc += static_cast<uint32_t>(diff * diff);
    }

#else
    // Scalar 4x unroll for baseline speed.
    size_t i = 0;
    const size_t limit = qty & ~static_cast<size_t>(3);
    for (; i < limit; i += 4) {
        int d0 = static_cast<int>(pVect1[i]) - static_cast<int>(pVect2[i]);
        int d1 = static_cast<int>(pVect1[i + 1]) - static_cast<int>(pVect2[i + 1]);
        int d2 = static_cast<int>(pVect1[i + 2]) - static_cast<int>(pVect2[i + 2]);
        int d3 = static_cast<int>(pVect1[i + 3]) - static_cast<int>(pVect2[i + 3]);
        acc += static_cast<uint32_t>(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
    }
    for (; i < qty; ++i) {
        const int diff = static_cast<int>(pVect1[i]) - static_cast<int>(pVect2[i]);
        acc += static_cast<uint32_t>(diff * diff);
    }
#endif

    return static_cast<float>(acc);
}

class L2SpaceUint8 final : public SpaceInterface<float> {
public:
    explicit L2SpaceUint8(size_t dim)
        : dim_(dim), data_size_(dim * sizeof(uint8_t)) {}

    size_t get_data_size() override {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() override {
        return L2SqrUint8;
    }

    void *get_dist_func_param() override {
        return &dim_;
    }

private:
    size_t dim_;
    size_t data_size_;
};

}  // namespace hnswlib
