#!/usr/bin/env bash
# ONNX → TensorRT 엔진 (FP16 기본, --int8 시 보정 캐시 사용). 실행: bash deploy/jetson/build_trt.sh deploy/jetson/models [--int8]
set -e
DIR=${1:-deploy/jetson/models}
INT8=${2:-}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
for onnx in "$DIR"/classifier_*.onnx "$DIR"/detector_*.onnx; do
  [ -f "$onnx" ] || continue
  base=$(basename "$onnx" .onnx)
  if [ "$INT8" = "--int8" ] && [ -f "$DIR/calib.cache" ]; then
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$DIR/${base}_int8.engine" --int8 --fp16 --calib="$DIR/calib.cache" --workspace=2048 --verbose=false
  else
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$DIR/${base}_fp16.engine" --fp16 --workspace=2048 --verbose=false
  fi
  echo "built: $base"
done
echo "엔진 목록:"; ls -la "$DIR"/*.engine
