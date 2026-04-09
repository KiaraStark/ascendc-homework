#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os

import numpy as np


def logsumexp(x, dims, keep_dim=False):
    x32 = x.astype(np.float32)
    dims = tuple(dims)
    x_max = np.max(x32, axis=dims, keepdims=True)
    y = np.log(np.sum(np.exp(x32 - x_max), axis=dims, keepdims=True)) + x_max
    if not keep_dim:
        y = np.squeeze(y, axis=dims)
    return y


def gen_data(dtype="fp16", dims=(0,), keep_dim=False, shape=(8, 16, 32), seed=0):
    np.random.seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    x = np.random.uniform(-8.0, 8.0, shape)
    x = x.astype(np.float32 if dtype == "fp32" else np.float16)
    golden = logsumexp(x, dims, keep_dim=keep_dim).astype(x.dtype)

    x.tofile(os.path.join(input_dir, "input_x.bin"))
    golden.tofile(os.path.join(output_dir, "golden.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LogSumExp test data")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--dim", type=int, nargs="+", default=[0])
    parser.add_argument("--keep_dim", action="store_true")
    parser.add_argument("--shape", type=int, nargs=3, default=[8, 16, 32])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    gen_data(args.dtype, tuple(args.dim), args.keep_dim, tuple(args.shape), args.seed)