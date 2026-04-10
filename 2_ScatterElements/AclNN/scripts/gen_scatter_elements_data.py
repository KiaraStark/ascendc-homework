#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os

import numpy as np


def make_value(shape, dtype):
    if dtype == "fp32":
        return np.random.uniform(-5.0, 5.0, shape).astype(np.float32)
    if dtype == "fp16":
        return np.random.uniform(-5.0, 5.0, shape).astype(np.float16)
    if dtype == "int32":
        return np.random.randint(-10, 10, shape, dtype=np.int32)
    if dtype == "uint8":
        return np.random.randint(0, 255, shape, dtype=np.uint8)
    raise ValueError(f"Unsupported dtype: {dtype}")


def scatter_elements(var, indices, updates, axis=0, reduce=None):
    out = var.copy()
    axis = axis % out.ndim

    for idx in np.ndindex(indices.shape):
        dst = list(idx)
        dst[axis] = int(indices[idx])
        dst = tuple(dst)

        if reduce is None or reduce == "none":
            out[dst] = updates[idx]
        elif reduce == "add":
            out[dst] = out[dst] + updates[idx]
        elif reduce == "multiply":
            out[dst] = out[dst] * updates[idx]
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")

    return out


def gen_data(dtype="fp16", axis=1, reduce=None, shape=(8, 16, 32), seed=0):
    np.random.seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_dir = os.path.join(project_dir, "input")
    output_dir = os.path.join(project_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    var = make_value(shape, dtype)
    updates = make_value(shape, dtype)
    indices = np.random.randint(0, shape[axis], shape, dtype=np.int32)
    golden = scatter_elements(var, indices, updates, axis=axis, reduce=reduce)

    var.tofile(os.path.join(input_dir, "var.bin"))
    indices.tofile(os.path.join(input_dir, "indices.bin"))
    updates.tofile(os.path.join(input_dir, "updates.bin"))
    golden.tofile(os.path.join(output_dir, "golden.bin"))


def main():
    parser = argparse.ArgumentParser(description="Generate ScatterElements test data")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "int32", "uint8"], default="fp16")
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--reduce", choices=["none", "add", "multiply"], default="none")
    parser.add_argument("--shape", type=int, nargs=3, default=[8, 16, 32])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.axis < -len(args.shape) or args.axis >= len(args.shape):
        raise ValueError(f"axis out of range: axis={args.axis}, rank={len(args.shape)}")

    reduce = None if args.reduce == "none" else args.reduce
    gen_data(args.dtype, args.axis, reduce, tuple(args.shape), args.seed)


if __name__ == "__main__":
    main()
