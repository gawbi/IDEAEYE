"""Jetson 에서 TensorRT 엔진 지연 벤치마크. python deploy/jetson/bench_jetson.py --engine <classifier_fp16.engine> [--n 200] [--meta classifier_meta.json]

pycuda + tensorrt 파이썬 바인딩 사용. 결과: 배치당/계란당 p50·p95 ms, JSON 저장.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    outs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    shape = tuple(engine.get_tensor_shape(inp))
    batch = shape[0]
    h_in = np.random.rand(*shape).astype(np.float32)
    d_in = cuda.mem_alloc(h_in.nbytes)
    d_outs = {o: cuda.mem_alloc(int(np.prod(engine.get_tensor_shape(o))) * 4) for o in outs}
    ctx.set_tensor_address(inp, int(d_in))
    for o, d in d_outs.items():
        ctx.set_tensor_address(o, int(d))
    stream = cuda.Stream()
    for _ in range(10):  # warm-up
        cuda.memcpy_htod_async(d_in, h_in, stream); ctx.execute_async_v3(stream.handle); stream.synchronize()
    ts = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_in, h_in, stream); ctx.execute_async_v3(stream.handle); stream.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    res = {"engine": args.engine, "input_shape": shape, "batch": batch, "ms_per_batch": {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max())},
           "ms_per_image": {"p50": float(np.percentile(a, 50) / batch), "p95": float(np.percentile(a, 95) / batch)}}
    try:
        res["power_mode"] = Path("/etc/nvpmodel.conf").exists()
    except Exception:
        pass
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
