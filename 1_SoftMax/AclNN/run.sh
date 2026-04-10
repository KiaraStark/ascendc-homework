#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/output"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

DTYPE="both"
REBUILD="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --rebuild)
            REBUILD="1"
            shift
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: ./run.sh [--dtype fp32|fp16|both] [--rebuild]

Options:
  --dtype   Data type to test. Default: both
  --rebuild Clean build directory before compiling.
USAGE
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$DTYPE" != "fp32" && "$DTYPE" != "fp16" && "$DTYPE" != "both" ]]; then
    echo "[ERROR] --dtype must be fp32, fp16 or both"
    exit 1
fi

if [[ -z "${ASCEND_HOME_DIR:-}" ]]; then
    if [[ -d "$HOME/Ascend/ascend-toolkit/latest" ]]; then
        ASCEND_HOME_DIR="$HOME/Ascend/ascend-toolkit/latest"
    else
        ASCEND_HOME_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    fi
fi

if [[ ! -f "$ASCEND_HOME_DIR/bin/setenv.bash" ]]; then
    echo "[ERROR] Cannot find setenv.bash in $ASCEND_HOME_DIR/bin"
    exit 1
fi

# shellcheck disable=SC1090
source "$ASCEND_HOME_DIR/bin/setenv.bash"

export DDK_PATH="$ASCEND_HOME_DIR"
arch=$(uname -m)
export NPU_HOST_LIB="$ASCEND_HOME_DIR/${arch}-linux/lib64"

echo "[INFO] ASCEND_HOME_DIR=$ASCEND_HOME_DIR"

do_build() {
    if [[ "$REBUILD" == "1" ]]; then
        echo "[INFO] Rebuild enabled, cleaning $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR"
    cmake --build "$BUILD_DIR" -j"$(nproc)"
}

run_single_dtype() {
    local dtype="$1"
    echo "[INFO] Running dtype=${dtype}"

    python3 "$SCRIPTS_DIR/gen_softmax_data.py" --dtype "$dtype"

    (
        cd "$OUTPUT_DIR"
        ./execute_softmax_aclnn --dtype "$dtype"
    )

    cp "$SCRIPTS_DIR/output/output.bin" "$SCRIPTS_DIR/output/output_${dtype}.bin"

    python3 "$SCRIPTS_DIR/verify_result.py" \
        "$SCRIPTS_DIR/output/output.bin" \
        "$SCRIPTS_DIR/output/golden.bin" \
        --dtype "$dtype"
}

do_build

if [[ "$DTYPE" == "both" ]]; then
    run_single_dtype fp32
    run_single_dtype fp16
else
    run_single_dtype "$DTYPE"
fi

echo "[INFO] Softmax ACLNN pipeline finished successfully"
