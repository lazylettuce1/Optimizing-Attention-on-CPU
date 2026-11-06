/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Placeholder: replace the body of attention() with your implementation.
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
#ifndef CACHE_ROWS
#define CACHE_ROWS 32
#endif
#ifndef CACHE_KEYS
#define CACHE_KEYS 64
#endif
#define REGISTER_KEYS 16

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *scores = malloc((size_t)N * N * sizeof(float));
    float *KT = malloc((size_t)N * d * sizeof(float));
    if (!scores || !KT) {
        fprintf(stderr, "attention: scratch allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < d; ++k)
            KT[(size_t)k * N + j] = K[(size_t)j * d + k];

    /* Cache tiles reuse a key panel across nearby queries. Four rows of
     * independent accumulators avoid repeatedly loading partial scores. */
    for (int first = 0; first < N; first += CACHE_ROWS) {
        int row_end = first + CACHE_ROWS < N ? first + CACHE_ROWS : N;
        for (int key = 0; key < N; key += CACHE_KEYS) {
            int key_end = key + CACHE_KEYS < N ? key + CACHE_KEYS : N;
            int i = first;
            for (; i + 3 < row_end; i += 4)
                for (int j = key; j < key_end; j += REGISTER_KEYS) {
                    int columns = key_end - j < REGISTER_KEYS ? key_end - j : REGISTER_KEYS;
                    float a0[REGISTER_KEYS] = {0}, a1[REGISTER_KEYS] = {0};
                    float a2[REGISTER_KEYS] = {0}, a3[REGISTER_KEYS] = {0};
                    for (int k = 0; k < d; ++k) {
                        const float *keys = KT + (size_t)k * N + j;
                        float q0 = Q[(size_t)i * d + k];
                        float q1 = Q[(size_t)(i + 1) * d + k];
                        float q2 = Q[(size_t)(i + 2) * d + k];
                        float q3 = Q[(size_t)(i + 3) * d + k];
                        for (int lane = 0; lane < columns; ++lane) {
                            a0[lane] += q0 * keys[lane];
                            a1[lane] += q1 * keys[lane];
                            a2[lane] += q2 * keys[lane];
                            a3[lane] += q3 * keys[lane];
                        }
                    }
                    for (int lane = 0; lane < columns; ++lane) {
                        scores[(size_t)i * N + j + lane] = a0[lane] * scale;
                        scores[(size_t)(i + 1) * N + j + lane] = a1[lane] * scale;
                        scores[(size_t)(i + 2) * N + j + lane] = a2[lane] * scale;
                        scores[(size_t)(i + 3) * N + j + lane] = a3[lane] * scale;
                    }
                }
            for (; i < row_end; ++i)
                for (int j = key; j < key_end; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < d; ++k)
                        sum += Q[(size_t)i * d + k] * KT[(size_t)k * N + j];
                    scores[(size_t)i * N + j] = sum * scale;
                }
        }
    }

    for (int i = 0; i < N; ++i) {
        float *row = scores + (size_t)i * N;
        float maximum = row[0];
        for (int j = 1; j < N; ++j)
            if (row[j] > maximum) maximum = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            row[j] = expf(row[j] - maximum);
            sum += row[j];
        }
        for (int j = 0; j < N; ++j)
            row[j] /= sum;
    }

    memset(O, 0, (size_t)N * d * sizeof(float));
    for (int first = 0; first < N; first += CACHE_ROWS) {
        int row_end = first + CACHE_ROWS < N ? first + CACHE_ROWS : N;
        for (int key = 0; key < N; key += CACHE_KEYS) {
            int key_end = key + CACHE_KEYS < N ? key + CACHE_KEYS : N;
            for (int i = first; i < row_end; ++i) {
                float *output = O + (size_t)i * d;
                for (int j = key; j < key_end; ++j) {
                    float probability = scores[(size_t)i * N + j];
                    const float *values = V + (size_t)j * d;
                    for (int k = 0; k < d; ++k)
                        output[k] += probability * values[k];
                }
            }
        }
    }
    free(KT);
    free(scores);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
