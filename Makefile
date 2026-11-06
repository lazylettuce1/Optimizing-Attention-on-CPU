# Override CC/CFLAGS on the command line to reproduce compiler experiments.
CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
OMP = -fopenmp
PYTHON = python3
INPUTS = $(sort $(wildcard input/*.bin))
VARIANTS = naive loops tiled fused best omp
RUN =

.DEFAULT_GOAL := all
.PHONY: all compile $(VARIANTS) check

all: $(VARIANTS)
compile: $(addprefix build/,$(VARIANTS))

build:
	mkdir -p $@

build/omp: src/attn_omp.c src/attention_kernel.h | build
	$(CC) $(CFLAGS) $(OMP) -o $@ $< -lm

build/best: src/attn_best.c src/attention_kernel.h | build
	$(CC) $(CFLAGS) -o $@ $< -lm

build/%: src/attn_%.c | build
	$(CC) $(CFLAGS) -o $@ $< -lm

$(VARIANTS): %: build/%
	@mkdir -p output
	@set -e; for f in $(INPUTS); do \
		$(RUN) ./build/$@ "$$f" "output/$@_$${f##*/}"; \
	done

# Missing files and numerical failures return nonzero. Does not rerun timings.
check:
	@$(PYTHON) -c "$$CHECK"

# Keep checking available in the submission without external Python files.
define CHECK
"""Check complete, finite FP32 outputs using the handout's error metric."""

import argparse
from array import array
from pathlib import Path
import math
import struct
import sys


def load_output(path, shape=None):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path}: missing header")
    n, d = struct.unpack_from("<ii", data)
    if n <= 0 or d <= 0 or len(data) != 8 + 4 * n * d:
        raise ValueError(f"{path}: invalid dimensions or payload length")
    if shape is not None and (n, d) != shape:
        raise ValueError(f"{path}: dimensions {(n, d)} != {shape}")
    values = array("f")
    values.frombytes(data[8:])
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: nonfinite output")
    return (n, d), values


def relative_error(reference, actual):
    if len(reference) != len(actual):
        raise ValueError("output lengths differ")
    scale = max(abs(value) for value in reference)
    difference = max(abs(a - b) for a, b in zip(actual, reference))
    return difference / scale if scale else (0.0 if difference == 0 else math.inf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--inputs", type=Path, default=Path("input"))
    parser.add_argument("--variants", nargs="+", default=["loops", "tiled", "fused", "best", "omp"])
    args = parser.parse_args()
    inputs = sorted(args.inputs.glob("*.bin"))
    if not inputs:
        parser.error("no input files")
    failed = False
    for input_path in inputs:
        try:
            with input_path.open("rb") as stream:
                shape = struct.unpack("<ii", stream.read(8))
            _, reference = load_output(args.output / f"naive_{input_path.name}", shape)
            for variant in args.variants:
                path = args.output / f"{variant}_{input_path.name}"
                try:
                    _, actual = load_output(path, shape)
                    error = relative_error(reference, actual)
                    passed = error <= 1e-4
                    print(f"{input_path.name:24} {variant:8} {'PASS' if passed else 'FAIL'} error={error:.3e}")
                    failed |= not passed
                except (OSError, ValueError) as error:
                    print(f"FAIL: {error}", file=sys.stderr)
                    failed = True
        except (OSError, ValueError, struct.error) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
endef
export CHECK

# Local development targets are optional and excluded from the submission.
-include .dev/Makefile
