#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/output"
INPUT_DIR="$SCRIPT_DIR/input"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

DTYPE="fp16"
REDUCE="none"
AXIS="1"
SHAPE="8,16,32"
SEED="0"
REBUILD="0"

usage() {
    cat <<'USAGE'
Usage:
  ./run.sh [--dtype fp16|fp32|int32|uint8] [--reduce none|add|multiply] [--axis N]
           [--shape D0,D1,D2] [--seed N] [--rebuild]

Options:
  --dtype    Input/output dtype. Default: fp16
  --reduce   ScatterElements reduce mode. Default: none
  --axis     Scatter axis. Default: 1
  --shape    3D shape for var/indices/updates, comma-separated. Default: 8,16,32
  --seed     Random seed for data generation. Default: 0
  --rebuild  Remove build dir and rebuild.
  -h, --help Show this help.

Examples:
  ./run.sh --dtype fp32 --reduce add
  ./run.sh --dtype uint8 --reduce multiply --axis 2 --shape 4,8,16 --seed 2026
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --reduce)
            REDUCE="$2"
            shift 2
            ;;
        --axis)
            AXIS="$2"
            shift 2
            ;;
        --shape)
            SHAPE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --rebuild)
            REBUILD="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

case "$DTYPE" in
    fp16|fp32|int32|uint8) ;;
    *)
        echo "[ERROR] --dtype must be one of: fp16|fp32|int32|uint8"
        exit 1
        ;;
esac

case "$REDUCE" in
    none|add|multiply) ;;
    *)
        echo "[ERROR] --reduce must be one of: none|add|multiply"
        exit 1
        ;;
esac

if ! [[ "$AXIS" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] --axis must be an integer"
    exit 1
fi

IFS=',' read -r SH0 SH1 SH2 <<< "$SHAPE"
if [[ -z "${SH0:-}" || -z "${SH1:-}" || -z "${SH2:-}" ]]; then
    echo "[ERROR] --shape must be D0,D1,D2"
    exit 1
fi
if ! [[ "$SH0" =~ ^[0-9]+$ && "$SH1" =~ ^[0-9]+$ && "$SH2" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --shape values must be non-negative integers"
    exit 1
fi
if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] --seed must be an integer"
    exit 1
fi

if [[ -z "${ASCEND_HOME_DIR:-}" ]]; then
    if [[ -d "$HOME/Ascend/ascend-toolkit/latest" ]]; then
        ASCEND_HOME_DIR="$HOME/Ascend/ascend-toolkit/latest"
    else
        ASCEND_HOME_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    fi
fi

if [[ -f "$ASCEND_HOME_DIR/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1090
    source "$ASCEND_HOME_DIR/bin/setenv.bash"
elif [[ -f "$ASCEND_HOME_DIR/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "$ASCEND_HOME_DIR/set_env.sh"
else
    echo "[ERROR] Cannot find Ascend environment script under $ASCEND_HOME_DIR"
    exit 1
fi

ARCH=$(uname -m)
export DDK_PATH="$ASCEND_HOME_DIR"
export NPU_HOST_LIB="$ASCEND_HOME_DIR/${ARCH}-linux/lib64"

if [[ "$REBUILD" == "1" ]]; then
    echo "[INFO] Rebuild enabled, clean: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR" "$INPUT_DIR"

echo "[INFO] Build execute_scatter_elements_acl"
cmake -S "$SRC_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "[INFO] Generate data: dtype=$DTYPE reduce=$REDUCE axis=$AXIS shape=$SHAPE seed=$SEED"
python3 "$SCRIPTS_DIR/gen_scatter_elements_data.py" \
    --dtype "$DTYPE" \
    --reduce "$REDUCE" \
    --axis "$AXIS" \
    --shape "$SH0" "$SH1" "$SH2" \
    --seed "$SEED"

echo "[INFO] Run operator on NPU"
(
    cd "$SCRIPT_DIR"
    "$OUTPUT_DIR/execute_scatter_elements_acl" --dtype "$DTYPE" --axis "$AXIS" --reduce "$REDUCE"
)

echo "[INFO] Verify output"
python3 "$SCRIPTS_DIR/verify_result.py" \
    "$OUTPUT_DIR/output.bin" \
    "$OUTPUT_DIR/golden.bin" \
    --dtype "$DTYPE"

echo "[INFO] ScatterElements ACLNN pipeline finished successfully"
