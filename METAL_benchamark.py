import time
import numpy as np
from tinygrad import Tensor 
from tinygrad.dtype import dtypes


sz = 1024
A = np.random.rand(sz, sz).astype(np.float32)
B = np.random.rand(sz, sz).astype(np.float32)
E = np.zeros((sz, sz))

DEVICE = "METAL" 

tiny_tensor_A = Tensor(A, dtype=dtypes.float32, device=DEVICE)
tiny_tensor_B = Tensor(B, dtype=dtypes.float32, device=DEVICE)
st = time.perf_counter()
C = (tiny_tensor_A @ tiny_tensor_B).realize()
t = time.perf_counter() - st
flop = 2 * A.shape[0]**3
flops = flop / t

print(f'METAL TFLOPs: {flops / 1e12: .6f} | Time {t:.4f}s')

# Kernel from TinyGrad

# template<typename T, typename S, typename U> U __metal_wmma(T m, T n, U o) {
#   S a,b,c; a.thread_elements()[0] = m.x; a.thread_elements()[1] = m.y; b.thread_elements()[0] = n.x; b.thread_elements()[1] = n.y;
#   c.thread_elements()[0] = o.x; c.thread_elements()[1] = o.y; simdgroup_multiply_accumulate(c, a, b, c);
#   return U(c.thread_elements()[0], c.thread_elements()[1]);
# }
# kernel void r_32_8_4_2_2_4_2_128_8_2_4_4(device float* data0, const device float* data1, const device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
#   int gidx0 = gid.y; /* 32 */
#   int gidx1 = gid.x; /* 8 */
#   int lidx2 = lid.z; /* 4 */
#   int lidx3 = lid.y; /* 2 */
#   int lidx4 = lid.x; /* 16 */
#   float2 acc0 = float2(0.0f,0.0f);
#   float2 acc1 = float2(0.0f,0.0f);
#   float2 acc2 = float2(0.0f,0.0f);
#   float2 acc3 = float2(0.0f,0.0f);
#   float2 acc4 = float2(0.0f,0.0f);
#   float2 acc5 = float2(0.0f,0.0f);
#   float2 acc6 = float2(0.0f,0.0f);
#   float2 acc7 = float2(0.0f,0.0f);
#   float2 acc8 = float2(0.0f,0.0f);
#   float2 acc9 = float2(0.0f,0.0f);
#   float2 acc10 = float2(0.0f,0.0f);
#   float2 acc11 = float2(0.0f,0.0f);
#   float2 acc12 = float2(0.0f,0.0f);
#   float2 acc13 = float2(0.0f,0.0f);
#   float2 acc14 = float2(0.0f,0.0f);
#   float2 acc15 = float2(0.0f,0.0f);
#   int alu0 = (gidx0*32768);
#   int alu1 = (lidx3*4096);
#   int alu2 = (((lidx4/2)%4)*1024);
#   int alu3 = ((lidx4%2)*2);
#   int alu4 = ((lidx4/8)*4);
#   int alu5 = (gidx1*128);
#   int alu6 = (lidx2*32);
#   int alu7 = (alu0+alu5+alu6+alu1+alu4+alu2+alu3);
#   for (int ridx0 = 0; ridx0 < 128; ridx0++) {
#     threadgroup_barrier(mem_flags::mem_threadgroup);
#     int alu8 = (alu0+alu1+alu2+(ridx0*8)+alu3+alu4);
#     float2 val0 = (float2)(*((device float2*)(data1+alu8)));
#     float2 val1 = (float2)(*((device float2*)(data1+alu8+8192)));
#     float2 val2 = (float2)(*((device float2*)(data1+alu8+16384)));
#     float2 val3 = (float2)(*((device float2*)(data1+alu8+24576)));
#     int alu9 = (alu5+alu6+alu4+alu3+(ridx0*8192)+alu2+alu1);
#     float2 val4 = (float2)(*((device float2*)(data2+alu9)));
#     float2 val5 = (float2)(*((device float2*)(data2+alu9+8)));
#     float2 val6 = (float2)(*((device float2*)(data2+alu9+16)));
#     float2 val7 = (float2)(*((device float2*)(data2+alu9+24)));
#     float2 cast0 = float2((val0).x,(val0).y);
#     float2 cast1 = float2((val4).x,(val4).y);
#     float2 wmma0 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast0, cast1, float2((acc0).x,(acc0).y));
#     (acc0).x = (wmma0).x;
#     (acc0).y = (wmma0).y;
#     float2 cast2 = float2((val1).x,(val1).y);
#     float2 wmma1 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast2, cast1, float2((acc1).x,(acc1).y));
#     (acc1).x = (wmma1).x;
#     (acc1).y = (wmma1).y;
#     float2 cast3 = float2((val2).x,(val2).y);
#     float2 wmma2 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast3, cast1, float2((acc2).x,(acc2).y));
#     (acc2).x = (wmma2).x;
#     (acc2).y = (wmma2).y;
#     float2 cast4 = float2((val3).x,(val3).y);
#     float2 wmma3 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast4, cast1, float2((acc3).x,(acc3).y));
#     (acc3).x = (wmma3).x;
#     (acc3).y = (wmma3).y;
#     float2 cast5 = float2((val5).x,(val5).y);
#     float2 wmma4 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast0, cast5, float2((acc4).x,(acc4).y));
#     (acc4).x = (wmma4).x;
#     (acc4).y = (wmma4).y;
#     float2 wmma5 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast2, cast5, float2((acc5).x,(acc5).y));
#     (acc5).x = (wmma5).x;
#     (acc5).y = (wmma5).y;
#     float2 wmma6 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast3, cast5, float2((acc6).x,(acc6).y));
#     (acc6).x = (wmma6).x;
#     (acc6).y = (wmma6).y;
#     float2 wmma7 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast4, cast5, float2((acc7).x,(acc7).y));
#     (acc7).x = (wmma7).x;
#     (acc7).y = (wmma7).y;
#     float2 cast6 = float2((val6).x,(val6).y);
#     float2 wmma8 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast0, cast6, float2((acc8).x,(acc8).y));
#     (acc8).x = (wmma8).x;
#     (acc8).y = (wmma8).y;
#     float2 wmma9 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast2, cast6, float2((acc9).x,(acc9).y));
#     (acc9).x = (wmma9).x;
#     (acc9).y = (wmma9).y;
#     float2 wmma10 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast3, cast6, float2((acc10).x,(acc10).y));
#     (acc10).x = (wmma10).x;
#     (acc10).y = (wmma10).y;
#     float2 wmma11 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast4, cast6, float2((acc11).x,(acc11).y));
#     (acc11).x = (wmma11).x;
#     (acc11).y = (wmma11).y;
#     float2 cast7 = float2((val7).x,(val7).y);
#     float2 wmma12 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast0, cast7, float2((acc12).x,(acc12).y));
#     (acc12).x = (wmma12).x;
#     (acc12).y = (wmma12).y;
#     float2 wmma13 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast2, cast7, float2((acc13).x,(acc13).y));
#     (acc13).x = (wmma13).x;
#     (acc13).y = (wmma13).y;
#     float2 wmma14 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast3, cast7, float2((acc14).x,(acc14).y));
#     (acc14).x = (wmma14).x;
#     (acc14).y = (wmma14).y;
#     float2 wmma15 = __metal_wmma<float2,simdgroup_float8x8,float2>(cast4, cast7, float2((acc15).x,(acc15).y));
#     (acc15).x = (wmma15).x;
#     (acc15).y = (wmma15).y;
#   }
#   *((device float2*)(data0+alu7)) = (float2)float2((acc0).x,(acc0).y);
#   *((device float2*)(data0+alu7+8192)) = (float2)float2((acc1).x,(acc1).y);
#   *((device float2*)(data0+alu7+16384)) = (float2)float2((acc2).x,(acc2).y);
#   *((device float2*)(data0+alu7+24576)) = (float2)float2((acc3).x,(acc3).y);
#   *((device float2*)(data0+alu7+8)) = (float2)float2((acc4).x,(acc4).y);
#   *((device float2*)(data0+alu7+8200)) = (float2)float2((acc5).x,(acc5).y);
#   *((device float2*)(data0+alu7+16392)) = (float2)float2((acc6).x,(acc6).y);
#   *((device float2*)(data0+alu7+24584)) = (float2)float2((acc7).x,(acc7).y);
#   *((device float2*)(data0+alu7+16)) = (float2)float2((acc8).x,(acc8).y);
#   *((device float2*)(data0+alu7+8208)) = (float2)float2((acc9).x,(acc9).y);
#   *((device float2*)(data0+alu7+16400)) = (float2)float2((acc10).x,(acc10).y);
#   *((device float2*)(data0+alu7+24592)) = (float2)float2((acc11).x,(acc11).y);
#   *((device float2*)(data0+alu7+24)) = (float2)float2((acc12).x,(acc12).y);
#   *((device float2*)(data0+alu7+8216)) = (float2)float2((acc13).x,(acc13).y);
#   *((device float2*)(data0+alu7+16408)) = (float2)float2((acc14).x,(acc14).y);
#   *((device float2*)(data0+alu7+24600)) = (float2)float2((acc15).x,(acc15).y);
# }
