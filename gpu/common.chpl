// Chapel port of common.h at
// https://github.com/karpathy/llm.c/blob/master/dev/cuda/common.h

module Common {
  // ceil_div functionality already provided Math.divCeil
  // No need for cuda/cublas error checking in Chapel

  // Random utils are also provided by Chapel
  // For a general pattern in common.h of creating a random array,
  // we can use the following Chapel code:
  // var randArray : [1..N] real;
  // Random.fillRandom(randArray, min, max);

  // Testing and benchmarking utils
  proc validateResult(ref deviceResultOnHost : [?D] ?t, const ref cpuReference,
                      const name:string, const absTolerance=1e-4,
                      const verbose = false) {
    for i in D {
      // print the first few comparisons
      if verbose && (i < 5) {
          writeln(cpuReference[i], " ", deviceResultOnHost[i]);
      }

      if !isClose(deviceResultOnHost[i], cpuReference[i], absTolerance){
        writeln("Mismatch of ", name, " at ", i, ": ", cpuReference[i], " vs ",
                deviceResultOnHost[i]);
        halt(1);
      }
    }
  }


  // We cannot use the way it's done in common.h to pass the kernel and
  // kernelargs.
  // since the are generic since they themselves take arrays as arguments.
  // See https://github.com/chapel-lang/chapel/issues/23760
  // We calso cannot use the workaround described in the issue above
  // because our "shim" would refer to outer variables as arguments,
  // and again be un-capturable
  // For now we just don't use this function, and instead inline it's body
  // wherever we need to benchmark something
  inline proc benchmarkKernel(const repeats: int, kernelShim, kernelArgs ...) {
    use Time only stopwatch;
    var s : stopwatch;
    s.start();
    for 1..repeats {
      kernelShim((...kernelArgs));
    }
    s.stop();
    return s.elapsed()/repeats;
  }
}
