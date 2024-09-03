/*
Chapel port of gelu_forward.cu at
https://github.com/karpathy/llm.c/blob/master/dev/cuda/gelu_forward.cu

Kernels for gelu forward pass.

version 1 is naive port from CPU code to kernel
./residualForward --vers=1

*/

use Common;
import Math.{pi, twiceReciprPi, tanh};
import Math.sqrt;

config const useGpuId=0;
config const vers=1;
config const correctnessOnly=false;
param GELU_SCALING_FACTOR = sqrt(twiceReciprPi); // same as sqrt(2 / pi)
type eltType = real(32);


// CPU reference
proc geluForwardCpu(ref output : [?D] eltType, const input : [] eltType){
  for i in D {
    const x = input[i];
    const cube = 0.044715 * x * x * x;
    const cdf = 0.5 * (1.0 + tanh(GELU_SCALING_FACTOR * (x + cube)));
    output[i] = x * cdf;
  }
}

// GPU kernels
proc geluForward1(ref output : [?D] eltType, const input : [] eltType,
                  const blockSize : int){
  @assertOnGpu
  @gpu.blockSize(blockSize)
  foreach i in D {
    const x = input[i];
    const cube = 0.044715 * x * x * x;
    output[i] = 0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + cube)));
  }
}

proc geluForward(const kernelNum : int = vers, ref output : [] eltType,
                 const input : [] eltType,
                 const blockSize : int) : void {
  select kernelNum {
    when 1 do geluForward1(output, input, blockSize);
    otherwise {
      writeln("Invalid kernel number: ", kernelNum);
      halt(1);
    }
  }
}

proc main(){
  import Random;

  param B = 8, T = 1024, C = 768;
  param size = B * T * C;

  var output : [0..<size] eltType;
  var input : [0..<size] eltType;
  Random.fillRandom(input, min=-1.0, max=1.0);

  // First check the correctness of the kernel
  // Get the right answer from CPU
  geluForwardCpu(output, input);

  writeln("Using kernel ", vers);

  const blockSizes = [32, 64, 128, 256, 512, 1024];

  for blockSize in blockSizes {
    writeln("Checking block size ", blockSize);
    var outputCpu : output.type;

    on here.gpus[useGpuId] {
      var outputGpu : output.type;
      var inputGpu = input;
      geluForward(vers, outputGpu, inputGpu, blockSize);
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
        var inputGpu = input;
        // Due to issues with capturing fcf's we have inline benchmarkKernel
        // If we get that working we can replace the rest of the body
        // with the commented line below
        //elapsedTime = benchmarkKernel(repeatTimes, residualForward, vers,
        //                              outputGpu, inputGpu,
        //                              blockSize);
        use Time only stopwatch;
        var s : stopwatch;
        s.start();
        for 1..repeatTimes {
          geluForward(vers, outputGpu, inputGpu, blockSize);
        }
        s.stop();
        elapsedTime =  s.elapsed() * 1e3 /repeatTimes ; // Seconds to miliseconds
      }
      // napkin math: estimate the memory bandwidth achieved
      // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
      // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
      const memoryOps = B * T * C * 2 * numBytes(eltType);
      const memoryBandwidth = memoryOps / elapsedTime / 1e6;

      writef("block_size %4i | time %.4dr ms | bandwidth %.2dr GB/s\n",
              blockSize, elapsedTime, memoryBandwidth);
    }

}