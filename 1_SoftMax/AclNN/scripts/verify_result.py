#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import sys

import numpy as np


MINIMUM = 1e-10


def verify_result(real_path, golden_path, dtype):
    real_result = np.fromfile(real_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if real_result.size != golden.size:
        print(f"[ERROR] size mismatch: real={real_result.size}, golden={golden.size}")
        return False

    # Softmax fp16/fp32 tolerances
    loss = 1e-3 if dtype == np.float16 else 1e-4

    diff = np.abs(real_result - golden)
    deno = np.maximum(np.abs(real_result), np.abs(golden))
    result_atol = np.less_equal(diff, loss)
    result_rtol = np.less_equal(diff / np.add(deno, MINIMUM), loss)

    if (not result_rtol.all()) and (not result_atol.all()):
        bad_rtol = np.sum(result_rtol == False)
        bad_atol = np.sum(result_atol == False)
        if bad_rtol > real_result.size * loss and bad_atol > real_result.size * loss:
            print("[ERROR] result error")
            return False

    print("test pass")
    return True


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Verify Softmax output against golden")
    parser.add_argument("real", help="Path to output file, e.g. output/output.bin")
    parser.add_argument("golden", help="Path to golden file, e.g. output/golden.bin")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp32",
        help="Data type used to decode bin files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dtype = np.float16 if args.dtype == "fp16" else np.float32
    ok = verify_result(args.real, args.golden, dtype)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
