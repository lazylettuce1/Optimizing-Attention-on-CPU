# Optimising Attention on a CPU

CS6886 Assignment 5. Submission files are at the repository root:

```text
src/                        # six variants and their shared AVX2 kernel
input/                      # four original inputs
machine_spec.txt
results.csv
plots/roofline.png
plots/time_N1024_d64.png
plots/time_N2048_d64.png
plots/time_N4096_d64.png
plots/time_N4096_d128.png
Makefile
assignment_handout.pdf
report.pdf
```

```sh
make compile
OMP_NUM_THREADS=1 make all RUN='taskset -c 0'
make check
```

The Makefile contains the output checker; submission builds do not need
local helper scripts. GCC, OpenMP, libm and Python 3 are sufficient.

`attn_best.c` implements a CPU FlashAttention forward pass: compute a small
score tile, update the running softmax maximum/sum and weighted output, then
reuse the tile. Earlier output is rescaled when the running maximum increases;
normalization happens once after all keys. This follows the tiled online
softmax idea in [FlashAttention](https://arxiv.org/abs/2205.14135).

Best and OpenMP share `src/attention_kernel.h`. Named constants select 32 query
rows and 64 keys; score scratch is 8 KiB per worker, independent of N. The
padded transposed K is still O(N*d) storage. AVX2/FMA and the tested vector
exponential are reused; the supplied drivers and naive reference are unchanged.

At N=4096, d=64, serial FlashAttention takes 45.193 ms; OpenMP with
12 threads takes 8.357 ms. All measured outputs pass the assignment's
error tolerance. The previous implementation is archived in `.dev/experiments/`;
paired measurements are in `.dev/flash_comparison.csv`. FlashAttention is not
assumed faster on every size; the recorded comparison includes slowdowns.

Local development materials are preserved in Git-ignored `.dev/`: the
unchanged starter, scripts, raw measurements/logs, report sources, and
extra plots. See `.dev/README.md` for local commands. `.tex` files remain
ignored. Generated `build/`, `output/`, and `.venv/` are also ignored.
