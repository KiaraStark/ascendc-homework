#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import sys

import numpy as np


MINIMUM = 1e-10


def verify_result(real_path, golden_path, dtype):
    real = np.fromfile(real_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if real.size != golden.size:
        print(f"[ERROR] size mismatch: real={real.size}, golden={golden.size}")
        return False

    if np.issubdtype(dtype, np.integer):
        ok = np.array_equal(real, golden)
        if not ok:
            bad = np.count_nonzero(real != golden)
            print(f"[ERROR] integer mismatch count={bad}, total={real.size}")
            return False
        print("test pass")
        return True

    loss = 1e-3 if dtype == np.float16 else 1e-4
    diff = np.abs(real - golden)
    deno = np.maximum(np.abs(real), np.abs(golden))

    result_atol = np.less_equal(diff, loss)
    denominator = np.add(deno, MINIMUM)
    relative = np.divide(diff, denominator, out=np.zeros_like(diff, dtype=np.float32), where=denominator > 0)
    result_rtol = np.less_equal(relative, loss)

    if (not result_rtol.all()) and (not result_atol.all()):
        bad_rtol = np.sum(result_rtol == False)
        bad_atol = np.sum(result_atol == False)
        if bad_rtol > real.size * loss and bad_atol > real.size * loss:
            max_abs = float(np.max(diff)) if diff.size > 0 else 0.0
            print(f"[ERROR] float mismatch, bad_rtol={bad_rtol}, bad_atol={bad_atol}, max_abs={max_abs:.6e}")
            return False

    print("test pass")
    return True


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Verify ScatterElements output against golden")
    parser.add_argument("real", help="Path to output file, e.g. ./output/output.bin")
    parser.add_argument("golden", help="Path to golden file, e.g. ./output/golden.bin")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32", "int32", "uint8"],
        default="fp16",
        help="Data type used to decode bin files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dtype_map = {
        "fp16": np.float16,
        "fp32": np.float32,
        "int32": np.int32,
        "uint8": np.uint8,
    }
    ok = verify_result(args.real, args.golden, dtype_map[args.dtype])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
