/* CPU FlashAttention forward pass: tiled scores and online stable softmax.
 * Based on https://arxiv.org/abs/2205.14135; no GPU or external math library.
 * Shared by best and omp: identical arithmetic, different row scheduling.
 * AVX2 lanes span independent keys/features, avoiding horizontal reductions.
 */
#ifndef ATTENTION_KERNEL_H
#define ATTENTION_KERNEL_H

#include <immintrin.h>

#ifndef QUERY_BLOCK
#define QUERY_BLOCK 32
#endif
#ifndef KEY_BLOCK
#define KEY_BLOCK 64
#endif
#define KEY_PADDING 16

static float *allocate_floats(size_t count)
{
    float *buffer = malloc(count * sizeof(float));
    if (!buffer) {
        fprintf(stderr, "attention: scratch allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return buffer;
}

/* KT[k, j] makes a group of independent key scores contiguous. */
static void transpose_keys(const float *K, float *KT, int N, int d)
{
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < d; ++k)
            KT[(size_t)k * (N + KEY_PADDING) + j] = K[(size_t)j * d + k];
}

/* exp(x) for softmax's x <= 0. Range reduction keeps |r| <= ln(2)/2;
 * a degree-seven Taylor polynomial then has truncation error below 1e-8.
 * Values below -80 contribute less than 2e-35 each and are set to zero.
 */
static __m256 exp_negative(__m256 x)
{
    __m256 cutoff = _mm256_set1_ps(-80.0f);
    __m256 keep = _mm256_cmp_ps(x, cutoff, _CMP_GE_OQ);
    x = _mm256_max_ps(x, cutoff);
    __m256 exponent = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504089f)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    /* Split ln(2) reduces cancellation when the exponent is large. */
    __m256 r = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(-0.000212194440055f), r);
    __m256 polynomial = _mm256_set1_ps(1.0f / 5040.0f);
    polynomial = _mm256_fmadd_ps(polynomial, r, _mm256_set1_ps(1.0f / 720.0f));
    polynomial = _mm256_fmadd_ps(polynomial, r, _mm256_set1_ps(1.0f / 120.0f));
    polynomial = _mm256_fmadd_ps(polynomial, r, _mm256_set1_ps(1.0f / 24.0f));
    polynomial = _mm256_fmadd_ps(polynomial, r, _mm256_set1_ps(1.0f / 6.0f));
    polynomial = _mm256_fmadd_ps(polynomial, r, _mm256_set1_ps(0.5f));
    polynomial = _mm256_fmadd_ps(polynomial, r, _mm256_set1_ps(1.0f));
    polynomial = _mm256_fmadd_ps(polynomial, r, _mm256_set1_ps(1.0f));
    __m256i bits = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(exponent),
                                                   _mm256_set1_epi32(127)), 23);
    return _mm256_and_ps(keep, _mm256_mul_ps(polynomial, _mm256_castsi256_ps(bits)));
}

/* Merge a score tile into the running softmax statistics. O holds the
 * unnormalized weighted sum, so division is needed only after the last tile.
 * new_max = max(old_max, tile_max)
 * alpha = exp(old_max - new_max)
 * new_sum = alpha * old_sum + sum(exp(tile_scores - new_max))
 * new_O = alpha * old_O + exp(tile_scores - new_max) * tile_V
 */
static void update_softmax(float *scores, int columns, float scale,
                           float *running_max, float *running_sum, float *output, int d)
{
    __m256 factor = _mm256_set1_ps(scale);
    __m256 maxima = _mm256_set1_ps(*running_max);
    int j = 0;
    for (; j + 7 < columns; j += 8) {
        __m256 x = _mm256_mul_ps(_mm256_loadu_ps(scores + j), factor);
        _mm256_storeu_ps(scores + j, x);
        maxima = _mm256_max_ps(maxima, x);
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, maxima);
    float maximum = lanes[0];
    for (int lane = 1; lane < 8; ++lane)
        if (lanes[lane] > maximum) maximum = lanes[lane];
    for (int tail = j; tail < columns; ++tail) {
        scores[tail] *= scale;
        if (scores[tail] > maximum) maximum = scores[tail];
    }

    /* A larger maximum changes the scale of all earlier contributions. */
    if (maximum > *running_max) {
        float alpha = expf(*running_max - maximum);
        *running_sum *= alpha;
        for (int k = 0; k < d; ++k)
            output[k] *= alpha;
    }
    *running_max = maximum;
    __m256 sums = _mm256_setzero_ps();
    __m256 shift = _mm256_set1_ps(maximum);
    for (j = 0; j + 7 < columns; j += 8) {
        __m256 x = exp_negative(_mm256_sub_ps(_mm256_loadu_ps(scores + j), shift));
        _mm256_storeu_ps(scores + j, x);
        sums = _mm256_add_ps(sums, x);
    }
    _mm256_storeu_ps(lanes, sums);
    float sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane) sum += lanes[lane];
    for (; j < columns; ++j) {
        scores[j] = expf(scores[j] - maximum);
        sum += scores[j];
    }
    *running_sum += sum;
}

/* Four queries x sixteen keys: eight independent vector accumulators. */
static void score_four(const float *Q, const float *KT, float *S, int N, int d,
                       int key_begin, int key_end)
{
    int j = key_begin;
    for (; j + 15 < key_end; j += 16) {
        __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        for (int k = 0; k < d; ++k) {
            const float *keys = KT + (size_t)k * (N + KEY_PADDING) + j;
            __m256 x = _mm256_loadu_ps(keys), y = _mm256_loadu_ps(keys + 8);
            __m256 q = _mm256_set1_ps(Q[k]);
            a0 = _mm256_fmadd_ps(q, x, a0); b0 = _mm256_fmadd_ps(q, y, b0);
            q = _mm256_set1_ps(Q[d + k]);
            a1 = _mm256_fmadd_ps(q, x, a1); b1 = _mm256_fmadd_ps(q, y, b1);
            q = _mm256_set1_ps(Q[2 * (size_t)d + k]);
            a2 = _mm256_fmadd_ps(q, x, a2); b2 = _mm256_fmadd_ps(q, y, b2);
            q = _mm256_set1_ps(Q[3 * (size_t)d + k]);
            a3 = _mm256_fmadd_ps(q, x, a3); b3 = _mm256_fmadd_ps(q, y, b3);
        }
        _mm256_storeu_ps(S + (j - key_begin), a0); _mm256_storeu_ps(S + (j - key_begin) + 8, b0);
        _mm256_storeu_ps(S + (size_t)KEY_BLOCK + (j - key_begin), a1); _mm256_storeu_ps(S + (size_t)KEY_BLOCK + (j - key_begin) + 8, b1);
        _mm256_storeu_ps(S + 2 * (size_t)KEY_BLOCK + (j - key_begin), a2); _mm256_storeu_ps(S + 2 * (size_t)KEY_BLOCK + (j - key_begin) + 8, b2);
        _mm256_storeu_ps(S + 3 * (size_t)KEY_BLOCK + (j - key_begin), a3); _mm256_storeu_ps(S + 3 * (size_t)KEY_BLOCK + (j - key_begin) + 8, b3);
    }
    for (int row = 0; row < 4; ++row)
        for (int column = j; column < key_end; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < d; ++k)
                sum += Q[(size_t)row * d + k] * KT[(size_t)k * (N + KEY_PADDING) + column];
            S[(size_t)row * KEY_BLOCK + (column - key_begin)] = sum;
        }
}

/* Four probability rows reuse each V vector. Output stays in registers. */
static void output_four(const float *S, const float *V, float *O, int d,
                        int key_begin, int key_end)
{
    int k = 0;
    for (; k + 15 < d; k += 16) {
        __m256 a0 = _mm256_loadu_ps(O + k);
        __m256 b0 = _mm256_loadu_ps(O + k + 8);
        __m256 a1 = _mm256_loadu_ps(O + 1 * (size_t)d + k);
        __m256 b1 = _mm256_loadu_ps(O + 1 * (size_t)d + k + 8);
        __m256 a2 = _mm256_loadu_ps(O + 2 * (size_t)d + k);
        __m256 b2 = _mm256_loadu_ps(O + 2 * (size_t)d + k + 8);
        __m256 a3 = _mm256_loadu_ps(O + 3 * (size_t)d + k);
        __m256 b3 = _mm256_loadu_ps(O + 3 * (size_t)d + k + 8);
        for (int j = key_begin; j < key_end; ++j) {
            const float *values = V + (size_t)j * d + k;
            __m256 x = _mm256_loadu_ps(values), y = _mm256_loadu_ps(values + 8);
            __m256 p = _mm256_set1_ps(S[j - key_begin]);
            a0 = _mm256_fmadd_ps(p, x, a0); b0 = _mm256_fmadd_ps(p, y, b0);
            p = _mm256_set1_ps(S[(size_t)KEY_BLOCK + (j - key_begin)]);
            a1 = _mm256_fmadd_ps(p, x, a1); b1 = _mm256_fmadd_ps(p, y, b1);
            p = _mm256_set1_ps(S[2 * (size_t)KEY_BLOCK + (j - key_begin)]);
            a2 = _mm256_fmadd_ps(p, x, a2); b2 = _mm256_fmadd_ps(p, y, b2);
            p = _mm256_set1_ps(S[3 * (size_t)KEY_BLOCK + (j - key_begin)]);
            a3 = _mm256_fmadd_ps(p, x, a3); b3 = _mm256_fmadd_ps(p, y, b3);
        }
        _mm256_storeu_ps(O + k, a0); _mm256_storeu_ps(O + k + 8, b0);
        _mm256_storeu_ps(O + (size_t)d + k, a1); _mm256_storeu_ps(O + (size_t)d + k + 8, b1);
        _mm256_storeu_ps(O + 2 * (size_t)d + k, a2); _mm256_storeu_ps(O + 2 * (size_t)d + k + 8, b2);
        _mm256_storeu_ps(O + 3 * (size_t)d + k, a3); _mm256_storeu_ps(O + 3 * (size_t)d + k + 8, b3);
    }
    for (int row = 0; row < 4; ++row)
        for (int column = k; column < d; ++column) {
            float sum = O[(size_t)row * d + column];
            for (int j = key_begin; j < key_end; ++j)
                sum += S[(size_t)row * KEY_BLOCK + (j - key_begin)] * V[(size_t)j * d + column];
            O[(size_t)row * d + column] = sum;
        }
}

/* Consume each score tile immediately; scratch size is independent of N. */
static void attend_block(const float *Q, const float *KT, const float *V,
                         float *O, float *scores, int rows, int N, int d)
{
    float running_max[QUERY_BLOCK], running_sum[QUERY_BLOCK];
    for (int row = 0; row < rows; ++row) {
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0f;
    }
    memset(O, 0, (size_t)rows * d * sizeof(float));
    float scale = 1.0f / sqrtf((float)d);

    for (int key = 0; key < N; key += KEY_BLOCK) {
        int end = key + KEY_BLOCK < N ? key + KEY_BLOCK : N;
        int columns = end - key;
        int row = 0;
        for (; row + 3 < rows; row += 4)
            score_four(Q + (size_t)row * d, KT, scores + (size_t)row * KEY_BLOCK,
                       N, d, key, end);
        for (int r = row; r < rows; ++r)
            for (int j = key; j < end; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < d; ++k)
                    sum += Q[(size_t)r * d + k] * KT[(size_t)k * (N + KEY_PADDING) + j];
                scores[(size_t)r * KEY_BLOCK + (j - key)] = sum;
            }

        for (int r = 0; r < rows; ++r)
            update_softmax(scores + (size_t)r * KEY_BLOCK, columns, scale,
                           &running_max[r], &running_sum[r], O + (size_t)r * d, d);

        row = 0;
        for (; row + 3 < rows; row += 4)
            output_four(scores + (size_t)row * KEY_BLOCK, V, O + (size_t)row * d,
                        d, key, end);
        for (int r = row; r < rows; ++r)
            for (int j = key; j < end; ++j)
                for (int k = 0; k < d; ++k)
                    O[(size_t)r * d + k] += scores[(size_t)r * KEY_BLOCK + j - key]
                                            * V[(size_t)j * d + k];
    }
    for (int row = 0; row < rows; ++row) {
        float inverse = 1.0f / running_sum[row];
        for (int k = 0; k < d; ++k)
            O[(size_t)row * d + k] *= inverse;
    }
}

#endif
