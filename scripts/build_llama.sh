#!/bin/bash
# Clone llama.cpp at the pinned commit and build the tools the harness needs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT=cd959bae99ff4bed22bef2604f45241c09bbb8c8
SRC="$ROOT/third_party/llama.cpp"
if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
git -C "$SRC" checkout -q --detach "$COMMIT"
test "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT"
cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$SRC/build" -j "$(sysctl -n hw.perflevel0.physicalcpu)" \
  --target llama-server llama-perplexity llama-cli llama-bench llama-tokenize
echo "built llama.cpp $(git -C "$SRC" rev-parse HEAD)"
