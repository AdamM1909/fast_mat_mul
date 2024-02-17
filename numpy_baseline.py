import numpy as np
import time 

sz = 1024
A = np.random.rand(sz, sz).astype(np.float32)
B = np.random.rand(sz, sz).astype(np.float32)
E = np.zeros((sz, sz))

cnt = 1000
tms = []
for _ in range(cnt):
    # D = E @ E # flush cache ? 
    st = time.perf_counter()
    C = A @ B
    tms.append(time.perf_counter() - st)

mean_tms = sum(tms)/len(tms)
flop = 2 * A.shape[0]**3
flops = flop / mean_tms


print(f'Numpy TFLOPs: {flops / 1e12: .2f} | Time {mean_tms:.4f}s')


# print(np.__config__.show())
# print(np.show_config())

# Numpy: fp32
# 220-240 GLOPs with standard numpy
# reinstall numpy to use accelerate 1800GFLOPS 
# 96 GFLOPs with  OMP_NUM_THREADS=1
# 183 GFLOPs with  OMP_NUM_THREADS=2
# 240 GFLOPs with  OMP_NUM_THREADS=3
# 340 GFLOPs with  OMP_NUM_THREADS=4
# 600 GFLOPs with  OMP_NUM_THREADS=8

# Defualt number of threads is 3
# C++: 0.55 GFLOPs

