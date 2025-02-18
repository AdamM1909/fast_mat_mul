import numpy as np 
from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import lower_schedule, CompiledRunner
import time, pprint
from dataclasses import replace

# Video which looks at this  https://www.youtube.com/watch?v=c1UHi7dncvo

"""
Theoretcial limit on load store on mac M1 Max is 400GB/s
"""

"""
Standard noopt = 1 code. 
*** CLANG      1 new_reduce                                arg  2 mem  0.07 GB tm     15.68ms/    15.68ms (     1.07 GFLOPS    4.3|4.3     GB/s) ['sum']
"""
# new_src = """
# void new_reduce(float* restrict data0, float* restrict data1) {
#   float acc0 = 0.0f;
#   for (int ridx0 = 0; ridx0 < 16777216; ridx0++) {
#     float val0 = *(data1+ridx0);
#     acc0 = (acc0+val0);
#   }
#   *(data0+0) = acc0;
# }"""


"""
Standard code optimized by using float4 vector addition. 
*** CLANG      1 new_reduce                                arg  2 mem  0.07 GB tm     15.66ms/    15.66ms (     1.07 GFLOPS    4.3|4.3     GB/s) ['sum']
"""
# new_src = """
# typedef float float4 __attribute__((aligned(16),vector_size(16)));
# void r_4194304_4(float* restrict data0, float* restrict data1) {
#   float acc0 = 0.0f;
#   for (int ridx0 = 0; ridx0 < 4194304; ridx0++) {
#     float4 val0 = *((float4*)((data1+(ridx0<<2))));
#     acc0 = (acc0+val0[0]+val0[1]+val0[2]+val0[3]);
#   }
#   *(data0+0) = acc0;
# }
# """

"""
Init the float 4 accumulator and instead of reducig across the 4 diemension at each step, do once at the end. 
*** CLANG      1 new_reduce                                arg  2 mem  0.07 GB tm   3913.50us/     3.91ms (     4.29 GFLOPS   17.1|17.1    GB/s) ['sum']
"""
# new_src = """
# typedef float float4 __attribute__((aligned(16),vector_size(16)));
# void new_reduce(float* restrict data0, float* restrict data1) {
#   float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
#   for (int ridx0 = 0; ridx0 < 4194304; ridx0++) {
#     float4 val0 = *((float4*)((data1+(ridx0<<2))));
#     acc0 += val0;
#   }
#   *(data0+0) = acc0[0]+acc0[1]+acc0[2]+acc0[3];
# }
# """

"""
Now unroll the loop meaning 4 x 4 adds adds are done per load.  
*** CLANG      1 new_reduce                                arg  2 mem  0.07 GB tm   1102.42us/     1.10ms (    15.22 GFLOPS   60.9|60.9    GB/s) ['sum']
"""
# new_src = """
# typedef float float4 __attribute__((aligned(16),vector_size(16)));
# void new_reduce(float* restrict data0, float* restrict data1) {
#   float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
#   for (int ridx0 = 0; ridx0 < 4194304; ridx0+=4) {
#     float4 val0 = *((float4*)((data1+((ridx0+0)<<2))));
#     float4 val1 = *((float4*)((data1+((ridx0+1)<<2))));
#     float4 val2 = *((float4*)((data1+((ridx0+2)<<2))));
#     float4 val3 = *((float4*)((data1+((ridx0+3)<<2))));
#     acc0 += val0 + val1 + val2 + val3;
#   }
#   *(data0+0) = acc0[0]+acc0[1]+acc0[2]+acc0[3];
# }
# """

"""
Have two accumulators
*** CLANG      1 new_reduce                                arg  2 mem  0.07 GB tm   1102.42us/     1.10ms (    15.22 GFLOPS   60.9|60.9    GB/s) ['sum']
"""
new_src = """
typedef float float4 __attribute__((aligned(16),vector_size(16)));
void new_reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int ridx0 = 0; ridx0 < 4194304; ridx0+=4) {
    float4 val0 = *((float4*)((data1+((ridx0+0)<<2))));
    float4 val1 = *((float4*)((data1+((ridx0+1)<<2))));
    float4 val2 = *((float4*)((data1+((ridx0+2)<<2))));
    float4 val3 = *((float4*)((data1+((ridx0+3)<<2))));
    acc0 += val0 + val1;
    acc1 += val2 + val3;
  }
  *(data0+0) = acc0[0]+acc0[1]+acc0[2]+acc0[3] + acc1[0]+acc1[1]+acc1[2]+acc1[3];
}
"""


if __name__ == "__main__":
    np_array = np.random.default_rng().random((4096, 4096), dtype=np.float32)
    a = Tensor(np_array).realize() 
    with Context(SPLIT_REDUCEOP=0):
        GlobalCounters.reset()
        out = a.sum()
        sis = out.schedule()
    
        for i, ei in enumerate(lower_schedule(sis)):
            prg_spec = ei.prg.p
            prg_spec = replace(prg_spec, name='new_reduce', src=new_src)
            ei = replace(ei, prg=CompiledRunner(prg_spec))
            ei.run()
        
    np.testing.assert_allclose(out.item(), np_array.sum(), rtol=1e-4)    
