#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os

import numpy as np


def softmax(x, dim):
    x32 = x.astype(np.float32)
    x_max = np.max(x32, axis=dim, keepdims=True)
    exp_x = np.exp(x32 - x_max)
    sum_x = np.sum(exp_x, axis=dim, keepdims=True)
    return exp_x / sum_x


def gen_data(dtype="fp32", dim=-1, shape=(64, 256), seed=0):
    np.random.seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    x = np.random.uniform(-8.0, 8.0, shape)
    x = x.astype(np.float32 if dtype == "fp32" else np.float16)
    golden = softmax(x, dim).astype(x.dtype)

    x.tofile(os.path.join(input_dir, "input_x.bin"))
    golden.tofile(os.path.join(output_dir, "golden.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Softmax test data")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--dim", type=int, default=-1)
    parser.add_argument("--shape", type=int, nargs=2, default=[64, 256])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    gen_data(args.dtype, args.dim, tuple(args.shape), args.seed)