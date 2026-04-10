#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/output"
SCRIPTS_DIR="$SCRIPT_DIR/script"

DTYPE="fp16"
DIMS=("0")
KEEP_DIM="false"
SHAPE="8,16,32"
SEED="0"
REBUILD="0"

usage() {
    cat <<'USAGE'
Usage:
  ./run.sh [--dtype fp16|fp32] [--dim d0 d1 ...] [--keep_dim [true|false]]
           [--shape D0,D1,D2] [--seed N] [--rebuild]

Options:
  --dtype      Input/output dtype. Default: fp16
  --dim        Reduction dims (int64), one or more values. Default: 0
  --keep_dim   Keep reduced dimensions. Supports true|false|1|0.
               If used without value, it is true. Default: false
  --shape      3D shape for input x, comma-separated. Default: 8,16,32
  --seed       Random seed for data generation. Default: 0
  --rebuild    Remove build dir and rebuild.
  -h, --help   Show this help.

Examples:
  ./run.sh --dtype fp16
  ./run.sh --dtype fp32 --dim 0 2 --keep_dim true
  ./run.sh --dtype fp16 --dim -1 --shape 4,8,16 --seed 2026
USAGE
}

parse_bool() {
    local v="$1"
    case "$v" in
        1|true|True|TRUE) echo "true" ;;
        0|false|False|FALSE) echo "false" ;;
        *) return 1 ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --dim)
            shift
            DIMS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if ! [[ "$1" =~ ^-?[0-9]+$ ]]; then
                    echo "[ERROR] --dim expects int64 values, got: $1"
                    exit 1
                fi
                DIMS+=("$1")
                shift
            done
            if [[ ${#DIMS[@]} -eq 0 ]]; then
                echo "[ERROR] --dim requires at least one int64 value"
                exit 1
            fi
            ;;
        --keep_dim)
            if [[ $# -ge 2 && "$2" != --* ]]; then
                if ! KEEP_DIM=$(parse_bool "$2"); then
                    echo "[ERROR] Invalid --keep_dim value: $2"
                    exit 1
                fi
                shift 2
            else
                KEEP_DIM="true"
                shift
            fi
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
    fp16|fp32) ;;
    *)
        echo "[ERROR] --dtype must be one of: fp16|fp32"
        exit 1
        ;;
esac

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

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR" "$SCRIPTS_DIR/input" "$SCRIPTS_DIR/output"

echo "[INFO] Build execute_log_sum_exp_acl"
cmake -S "$SRC_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "[INFO] Generate data: dtype=$DTYPE dim=${DIMS[*]} keep_dim=$KEEP_DIM shape=$SHAPE seed=$SEED"
GEN_CMD=(
    python3 "$SCRIPTS_DIR/gen_logsumexp_data.py"
    --dtype "$DTYPE"
    --dim "${DIMS[@]}"
    --shape "$SH0" "$SH1" "$SH2"
    --seed "$SEED"
)
if [[ "$KEEP_DIM" == "true" ]]; then
    GEN_CMD+=(--keep_dim)
fi
"${GEN_CMD[@]}"

echo "[INFO] Run operator on NPU"
(
    cd "$SCRIPT_DIR"
    "$OUTPUT_DIR/execute_log_sum_exp_acl" --dtype "$DTYPE" --dim "${DIMS[@]}" --keep_dim "$KEEP_DIM"
)

echo "[INFO] Verify output"
python3 "$SCRIPTS_DIR/verify_result.py" \
    "$SCRIPTS_DIR/output/output.bin" \
    "$SCRIPTS_DIR/output/golden.bin" \
    --dtype "$DTYPE"

echo "[INFO] LogSumExp ACLNN pipeline finished successfully"
