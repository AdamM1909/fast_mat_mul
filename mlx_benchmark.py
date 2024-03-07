import mlx.core as mx

import numpy as np
import time 

sz = 1024
A = np.random.rand(sz, sz).astype(np.float32)
B = np.random.rand(sz, sz).astype(np.float32)
A_mx = mx.array(A)
B_mx = mx.array(B)


cnt = 1000
tms = []
for _ in range(cnt):
    # A = np.random.rand(sz, sz).astype(np.float32)
    # B = np.random.rand(sz, sz).astype(np.float32)
    # A_mx = mx.array(A)
    # B_mx = mx.array(B)
    st = time.perf_counter()
    C = A_mx @ B_mx
    mx.eval(C)
    tms.append(time.perf_counter() - st)

mean_tms = sum(tms)/len(tms)
flop = 2 * A.shape[0]**3
flops = flop / mean_tms


print(f'Numpy TFLOPs: {flops / 1e12: .2f} | Time {mean_tms:.4f}s')