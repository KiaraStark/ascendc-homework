import argparse
import os
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def softmax_py_float(x):
    """
    Compute softmax on the last axis.
    The computation is promoted to float32 for numeric stability.
    """
    x32 = x.astype(np.float32)
    orig_shape = x32.shape
    x_max = np.max(x32, axis=-1, keepdims=True)
    x_sub = x32 - x_max
    x_exp = np.exp(x_sub)
    x_sum = np.sum(x_exp, axis=-1, keepdims=True)
    out = x_exp / x_sum
    return out.reshape(orig_shape), x_max.reshape(orig_shape[0], 1), x_sum.reshape(orig_shape[0], 1)


def gen_golden_data_simple(dtype="both"):
    x_shape = (960, 960)
    workspace_shape = (1024,)

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if dtype in ("float32", "both"):
        x32 = np.random.uniform(-1, 1, x_shape).astype(np.float32)
        golden32, max32, sum32 = softmax_py_float(x32)

        output_max32 = np.zeros([x_shape[0], 8], dtype=np.float32) + max32
        output_sum32 = np.zeros([x_shape[0], 8], dtype=np.float32) + sum32
        workspace = np.zeros(workspace_shape, dtype=np.uint32)

        # Keep legacy filenames for existing verification flow.
        x32.tofile(os.path.join(INPUT_DIR, "input_x.bin"))
        workspace.tofile(os.path.join(INPUT_DIR, "workspace.bin"))
        golden32.astype(np.float32).tofile(os.path.join(OUTPUT_DIR, "golden.bin"))
        output_max32.tofile(os.path.join(OUTPUT_DIR, "golden_max.bin"))
        output_sum32.tofile(os.path.join(OUTPUT_DIR, "golden_sum.bin"))

        # Additional explicit fp32 files.
        x32.tofile(os.path.join(INPUT_DIR, "input_x_fp32.bin"))
        golden32.astype(np.float32).tofile(os.path.join(OUTPUT_DIR, "golden_fp32.bin"))

    if dtype in ("float16", "both"):
        x16 = np.random.uniform(-1, 1, x_shape).astype(np.float16)
        golden16_f32, _, _ = softmax_py_float(x16)

        # Save fp16 input and fp16/fp32 golden for flexible verification.
        x16.tofile(os.path.join(INPUT_DIR, "input_x_fp16.bin"))
        golden16_f32.astype(np.float16).tofile(os.path.join(OUTPUT_DIR, "golden_fp16.bin"))
        golden16_f32.astype(np.float32).tofile(os.path.join(OUTPUT_DIR, "golden_fp16_ref_fp32.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate softmax test data for float32/float16")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "both"],
        default="both",
        help="Which data type to generate",
    )
    args = parser.parse_args()

    gen_golden_data_simple(args.dtype)
