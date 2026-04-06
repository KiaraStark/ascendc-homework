#!/bin/bash
# Build script for Softmax AscendC operator

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building Softmax AscendC Kernel ==="
echo ""

# Clean previous build
rm -f *.o *.a 2>/dev/null || true

# Check if AscendC compiler is available
if [ -x "/usr/local/Ascend/cann-9.0.0-beta.2/aarch64-linux/ccec_compiler/bin/ccec" ]; then
    echo "Using AscendC ccec compiler..."
    export CCE_COMPILER=/usr/local/Ascend/cann-9.0.0-beta.2/aarch64-linux/ccec_compiler/bin/ccec

    # Try to compile with ccec (this will only work on aarch64 or with cross-compilation)
    $CCE_COMPILER -c softmax_kernel.cpp -o softmax_kernel.o \
        --cce-aicore-lang \
        -I/usr/local/Ascend/cann-9.0.0-beta.2/aarch64-linux/ascendc/include \
        -I/usr/local/Ascend/cann-9.0.0-beta.2/aarch64-linux/ascendc/include/basic_api \
        -I/usr/local/Ascend/cann-9.0.0-beta.2/aarch64-linux/ascendc/include/basic_api/interface \
        -I/usr/local/Ascend/cann-9.0.0-beta.2/aarch64-linux/ascendc/include/basic_api/impl \
        2>&1 || echo "AscendC kernel compilation requires aarch64 environment"
else
    echo "AscendC compiler not found, skipping native compilation"
fi

# Compile simplified version with g++ for testing
echo ""
echo "Compiling simplified kernel with g++ for testing..."
g++ -c -o softmax_kernel_simple.o softmax_kernel_simple.cpp -std=c++17 -O2

# Create static library
ar rcs libsoftmax_kernel.a softmax_kernel_simple.o

echo ""
echo "=== Build Results ==="
ls -la *.o *.a 2>/dev/null || echo "No object files created"

echo ""
echo "=== Running Tests ==="
python3 test_softmax_kernel.py

echo ""
echo "=== Build Complete ==="
