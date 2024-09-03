/*
Chapel port of residual_forward.cu at
https://github.com/karpathy/llm.c/blob/master/dev/cuda/residual_forward.cu

Kernels for residual forward pass.

version 1 is naive port from CPU code to kernel
./residualForward --vers=1
*/

use Common;

config const useGpuId=0;
config const vers=1;
config const correctnessOnly=false;
type eltType = real;

// CPU reference
proc residualForwardCpu(ref output : [?D] eltType, const inp1 : [] eltType,
                        const inp2 : [] eltType) {
  for i in D {
    output[i] = inp1[i] + inp2[i];
  }
}

// GPU kernels
proc residualForward1(ref output : [?D] eltType, const inp1 : [] eltType,
                      const inp2 : [] eltType, const blockSize : int) {
  @assertOnGpu
  @gpu.blockSize(blockSize)
  foreach i in D do
    output[i] = inp1[i] + inp2[i];
}

proc residualForward(const kernelNum : int = vers, ref output : [] eltType,
                     const inp1 : [] eltType, const inp2 : [] eltType,
                     const blockSize : int) : void {
  select kernelNum {
    when 1 do residualForward1(output, inp1, inp2, blockSize);
    otherwise {
      writeln("Invalid kernel number: ", kernelNum);
      halt(1);
    }
  }
}

proc main() {
  import Random;

  param B = 8, T = 1024, C = 768;
  param size = B * T * C;

  var output : [0..<size] eltType;
  var inp1 : [0..<size] eltType;
  var inp2 : [0..<size] eltType;
  Random.fillRandom(inp1, min=-1.0, max=1.0);
  Random.fillRandom(inp2, min=-1.0, max=1.0);

  // First check the correctness of the kernel
  // Get the right answer from CPU
  residualForwardCpu(output, inp1, inp2);

  const blockSizes = [32, 64, 128, 256, 512, 1024];
  writeln("Using kernel ", vers);

  for blockSize in blockSizes {
    writeln("Checking block size ", blockSize);
    var outputCpu : output.type;

    on here.gpus[useGpuId] {
      var outputGpu : output.type;
      var inp1Gpu = inp1;
      var inp2Gpu = inp2;
      residualForward(vers, outputGpu, inp1Gpu, inp2Gpu, blockSize);
      outputCpu = outputGpu;
    }

    validateResult(outputCpu, output, "out", (1e-5):eltType,
                   verbose=!correctnessOnly);
  }

  writeln("All results match. Starting benchmarks.");


  // Benchmark the kernel
  if !correctnessOnly then
    for blockSize in blockSizes {
      const repeatTimes = 1000;
      var elapsedTime : real;
      on here.gpus[useGpuId] {
        var outputGpu : output.type;
        var inp1Gpu = inp1;
        var inp2Gpu = inp2;
        // Due to issues with capturing fcf's we have inline benchmarkKernel
        // If we get that working we can replace the rest of the body
        // with the commented line below
        //elapsedTime = benchmarkKernel(repeatTimes, residualForward, vers,
        //                              outputGpu, inp1Gpu, inp2Gpu,
        //                              blockSize);
        use Time only stopwatch;
        var s : stopwatch;
        s.start();
        for 1..repeatTimes {
          residualForward(vers, outputGpu, inp1Gpu, inp2Gpu, blockSize);
        }
        s.stop();
        elapsedTime =  s.elapsed() * 1e3 /repeatTimes ; // Seconds to miliseconds
      }
      // napkin math: estimate the memory bandwidth achieved
      // for each (B,T,C) output element, we do 2 read and 1 write, 4 bytes each
      // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
      const memoryOps = B * T * C * 3 * numBytes(eltType);
      const memoryBandwidth = memoryOps / elapsedTime / 1e6;

      writef("block_size %4i | time %.4dr ms | bandwidth %.2dr GB/s\n",
              blockSize, elapsedTime, memoryBandwidth);
    }
}
