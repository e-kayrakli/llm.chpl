/*
   (C) Copyright 2024 Hewlett Packard Enterprise Development LP

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
   OTHER DEALINGS IN THE SOFTWARE.
*/

use CTypes;
use IO;
import Time;
import FileSystem;
import Reflection;
import Math;
import Types;

param GPT2_EOT = 50256:int(32);

record DataLoader {
  // hyperparameters
  var B: int(32);
  var T: int(32);
  // input handling and its state
  var tokens_file: file;
  var tokens_file_reader: fileReader(locking=false);
  var file_size: int;
  var current_position: int;


  // output memory
  const batchDom = {0..#(B*T+1)};
  var batch: [batchDom] int(32);
  const inputsDom = {0..#(B*T)};
  const targetsDom = {1..#(B*T)};

  // convenience variables
  var num_batches: int(32);
}

proc DataLoader.init(filename, B, T) {
  this.B = B;
  this.T = T;
  this.tokens_file = try! open(filename, ioMode.r);
  this.tokens_file_reader = this.tokens_file.reader(locking=false);
  this.file_size = this.tokens_file.size;
  if this.file_size < (B*T+1)*numBytes(int(32)) then
    halt("Error: file size is too small for the batch size and sequence ",
         "length\n");
  this.num_batches = (this.file_size / (B*T*numBytes(int(32)))):int(32);
}

proc ref DataLoader.nextBatch() {
    const B = this.B;
    const T = this.T;
    // if we are at the end of the file, loop back to the beginning
    if (this.current_position + (B*T+1) * numBytes(int(32)) > this.file_size) {
        this.current_position = 0;
    }
    // read the B*T+1 integers from the file into batch
    this.tokens_file_reader.seek(this.current_position..);
    this.tokens_file_reader.readBinary(this.batch);
    // advance the current position by B*T integers
    this.current_position += B*T * numBytes(int(32));
}

proc ref DataLoader.reset() {
  this.current_position = 0;
}

record GPT2Config {
  var max_seq_len: int(32); // max sequence length, e.g. 1024
  var vocab_size: int(32); // vocab size, e.g. 50257
  var num_layers: int(32); // number of layers, e.g. 12
  var num_heads: int(32); // number of heads in attention, e.g. 12
  var channels: int(32); // number of channels, e.g. 768
}

param NUM_PARAMETER_TENSORS = 16;
param NUM_ACTIVATION_TENSORS = 23;

class ParameterTensors {
  const V, C, maxT, L: int;
  param nonArrayArgs = 4+1; // 4 is what we have above, 1 is self

  var wte: [0..#(V*C)] real(32);
  var wpe: [0..#(maxT*C)] real(32);
  var ln1w: [0..#(L*C)] real(32);
  var ln1b: [0..#(L*C)] real(32);
  var qkvw: [0..#(L*3*C*C)] real(32);
  var qkvb: [0..#(L*3*C)] real(32);
  var attprojw: [0..#(L*C*C)] real(32);
  var attprojb: [0..#(L*C)] real(32);
  var ln2w: [0..#(L*C)] real(32);
  var ln2b: [0..#(L*C)] real(32);
  var fcw: [0..#(L*4*C*C)] real(32);
  var fcb: [0..#(L*4*C)] real(32);
  var fcprojw: [0..#(L*C*4*C)] real(32);
  var fcprojb: [0..#(L*C)] real(32);
  var lnfw: [0..#C] real(32);
  var lnfb: [0..#C] real(32);
}

proc ParameterTensors.numTotalParams() {
  var sum: int;
  for param i in nonArrayArgs..#NUM_PARAMETER_TENSORS {
    ref f = Reflection.getFieldRef(this, i);
    compilerAssert(isArray(f), "Expected a field to be an array ",
                               "did you change the struct?");
    sum += f.size;
  }
  return sum;
}

proc ParameterTensors.zeroAll() {
  for param i in nonArrayArgs..#NUM_PARAMETER_TENSORS {
    ref f = Reflection.getFieldRef(this, i);
    compilerAssert(isArray(f), "Expected a field to be an array ",
                               "did you change the struct?");
    f = 0;
  }
}

iter ParameterTensors.these() ref {
  for param i in nonArrayArgs..#NUM_PARAMETER_TENSORS {
    ref f = Reflection.getFieldRef(this, i);
    compilerAssert(isArray(f), "Expected a field to be an array ",
                               "did you change the struct?");
    for item in f do yield item;
  }
}

proc ParameterTensors.readFrom(reader) {
  for param i in nonArrayArgs..#NUM_PARAMETER_TENSORS {
    ref f = Reflection.getFieldRef(this, i);
    compilerAssert(isArray(f), "Expected a field to be an array ",
                               "did you change the struct?");
    reader.readBinary(f);
  }
}

proc encoder_forward(ref outp, const ref inp, const ref wte, const ref wpe,
                     B, T, C) {
  for b in 0..<B {
    for t in 0..<T {
      // seek to the output position in out[b,t,:]
      const outOff = b * T * C + t * C;
      // get the index of the token at inp[b, t]
      const ix = inp[b * T + t];
      // seek to the position in wte corresponding to the token
      const wteOff = ix * C;
      // seek to the position in wpe corresponding to the position
      const wpeOff = t * C;
      // add the two vectors and store the result in out[b,t,:]
      for (outpIdx, wteIdx, wpeIdx) in zip(outOff..#C, wteOff.., wpeOff..) {
        outp[outpIdx] = wte[wteIdx] + wpe[wpeIdx];
      }
    }
  }
}

proc layernorm_forward(ref outp, outpOff,
                       ref mean, meanOff,
                       ref rstd, rstdOff,
                       const ref inp, inpOff,
                       const ref weight, weightOff,
                       const ref bias, biasOff,
                       B, T, C) {

  const eps = 1e-5: real(32);
  for b in 0..<B {
    for t in 0..<T {
      // seek to the input position inp[b,t,:]
      const inpStart = inpOff + b * T * C + t * C;
      // calculate the mean
      var m = 0.0: real(32);
      for i in inpStart..#C {
        m += inp[i];
      }
      m = m/C;
      // calculate the variance (without any bias correction)
      var v = 0.0: real(32);
      for i in inpStart..#C {
        const inpShift = inp[i] - m;
        v += inpShift * inpShift;
      }
      v = v/C;
      // calculate the rstd
      const s = 1.0:real(32) / Math.sqrt(v + eps);
      // seek to the output position in out[b,t,:]
      const outpStart = outpOff + b * T * C + t * C;
      for (outpIdx, inpIdx, weightIdx, biasIdx) in zip(outpStart..#C,
                                                       inpStart..,
                                                       weightOff..,
                                                       biasOff..) {
        const n = (s * (inp[inpIdx] - m)); // normalized output
        const o = n * weight[weightIdx] + bias[biasIdx]; // scale and shift it
        outp[outpIdx] = o; // write
      }
      // cache the mean and rstd for the backward pass later
      mean[meanOff+(b*T+t)] = m;
      rstd[rstdOff+(b*T+t)] = s;
    }
  }
}

proc matmul_forward_help(ref outp, outpOff,
                         const ref inp, inpOff,
                         const ref weight, weightOff,
                         const ref bias, biasOff,
                         B, T, C, OC,
                         hasBias: bool) {
  // most of the running time is spent here and in matmul_backward
  // OC is short for "output channels"
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  // out will be (B,T,OC)
  forall (b,t) in {0..<B, 0..<T} {
    const outpStart = outpOff + b * T * OC + t * OC ;
    const inpStart = inpOff + b * T * C + t * C;
    for (outpIdx, biasIdx, o) in zip(outpStart..#OC, biasOff.., 0..) {
      var val = if hasBias then bias[biasIdx] else 0.0:real(32);
      const weightStart = weightOff + o*C;
      for (inpIdx, weightIdx) in zip(inpStart..#C, weightStart..) {
        val += inp[inpIdx] * weight[weightIdx];
      }
      outp[outpIdx] = val;
    }
  }
}

inline proc matmul_forward(ref outp, outpOff,
                    const ref inp, inpOff,
                    const ref weight, weightOff,
                    B, T, C, OC) {
  const bias = [0];
  matmul_forward_help(outp, outpOff,
                      inp, inpOff,
                      weight, weightOff,
                      bias, 0,
                      B, T, C, OC,
                      hasBias=false);
}

inline proc matmul_forward(ref outp, outpOff,
                    const ref inp, inpOff,
                    const ref weight, weightOff,
                    const ref bias, biasOff,
                    B, T, C, OC) {
  matmul_forward_help(outp, outpOff,
                      inp, inpOff,
                      weight, weightOff,
                      bias, biasOff,
                      B, T, C, OC,
                      hasBias=true);
}

proc attention_forward(ref outp, outpOff,
                       ref preatt, preattOff,
                       ref att, attOff,
                       const ref inp, inpOff,
                       B, T, C, NH) {
  // input is (B, T, 3C) Q,K,V
  // preatt, att are (B, NH, T, T)
  // output is (B, T, C)
  const C3 = C*3;
  const hs = C / NH; // head size
  const scale = 1.0 / Math.sqrt(hs);

  /*#pragma omp parallel for collapse(3)*/
  forall (b, t, h) in {0..<B, 0..<T, 0..<NH} {
    const queryStart = inpOff + b * T * C3 + t * C3 + h * hs;

    const preattStart = preattOff + b*NH*T*T + h*T*T + t*T;

    const attStart = attOff + b*NH*T*T + h*T*T + t*T;

    // pass 1: calculate query dot key and maxval
    var maxval = -10000.0:real(32); // TODO something better
    for (t2, preattIdx) in zip(0..t, preattStart..) {
      const keyStart = inpOff + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

      // (query_t) dot (key_t2)
      var val = 0.0: real(32);
      for (queryIdx, keyIdx) in zip(queryStart..#hs, keyStart..) {
        val += inp[queryIdx] * inp[keyIdx];
      }
      val *= scale;
      if val > maxval {
        maxval = val;
      }

      preatt[preattIdx] = val;
    }

    // pass 2: calculate the exp and keep track of sum
    var expsum = 0.0: real(32);
    for (preattIdx, attIdx) in zip(preattStart..#(t+1), attStart..) {
      const expv = Math.exp(preatt[preattIdx] - maxval);
      expsum += expv;
      att[attIdx] = expv;
    }
    var expsum_inv = (if expsum==0.0 then 0.0 else 1.0/expsum):real(32);

    // pass 3: normalize to get the softmax
    for (t2, attIdx) in zip(0..<T, attStart..) {
      if t2 <= t {
        att[attIdx] *= expsum_inv;
      } else {
        // causal attention mask. not strictly necessary to set to zero here
        // only doing this explicitly for debugging and checking to PyTorch
        att[attIdx] = 0.0;
      }
    }

    // pass 4: accumulate weighted values into the output of attention
    const outpStart = outpOff + b * T * C + t * C + h * hs;
    for outpIdx in outpStart..#hs { outp[outpIdx] = 0.0; }
    for (t2, attIdx) in zip(0..t, attStart..) {
      const inpStart = inpOff + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
      const att_btht2 = att[attIdx];
      for (outpIdx, inpIdx) in zip(outpStart..#hs, inpStart..) {
        outp[outpIdx] += att_btht2 * inp[inpIdx];
      }
    }
  }
}


proc residual_forward(ref outp, outpOff,
                      const ref inp1, inp1Off,
                      const ref inp2, inp2Off,
                      N) {
  for (inp1Idx, outpIdx, inp2Idx) in zip(inp1Off..#N, outpOff.., inp2Off..) {
    outp[outpIdx] = inp1[inp1Idx] + inp2[inp2Idx];
  }
}

proc gelu_forward(ref outp, outpOff,
                  const ref inp, inpOff,
                  N) {
    const s = Math.sqrt(2.0:real(32) / Math.pi);
    for (inpIdx, outpIdx) in zip(inpOff..#N, outpOff..) {
        const x = inp[inpIdx];
        const cube = 0.044715:real(32) * x * x * x;
        outp[outpIdx] = (0.5 * x * (1.0 + Math.tanh(s * (x + cube)))):real(32);
    }
}

// Engin: offsets are currently unused
proc softmax_forward(ref probs, probsOff,
                     const ref logits, logitsOff,
                     B, T, V) {
    // output: probs are (B,T,V) of the probabilities
    // input: logits is (B,T,V) of the unnormalized log probabilities
  forall (b,t) in {0..<B, 0..<T} {
    // probs <- softmax(logits)
    const logitsStart = logitsOff + b * T * V + t * V;
    const probsStart = probsOff + b * T * V + t * V;

    var maxval = -10000.0:real(32); // TODO something better
    for i in logitsStart..#V {
      if (logits[i] > maxval) {
        maxval = logits[i];
      }
    }

    var sum = 0.0:real(32);
    for (probsIdx, logitsIdx) in zip(probsStart..#V, logitsStart..) {
      probs[probsIdx] = Math.exp(logits[logitsIdx] - maxval);
      sum += probs[probsIdx];
    }

    for probsIdx in probsStart..#V {
      probs[probsIdx] /= sum;
    }
  }
}

proc crossentropy_forward(ref losses,
                          const ref probs,
                          const ref targets,
                          B, T, V) {
  // output: losses is (B,T) of the individual losses at each position
  // input: probs are (B,T,V) of the probabilities
  // input: targets is (B,T) of integers giving the correct index in logits
  for b in 0..<B {
    for t in 0..<T {
      // loss = -log(probs[target])
      const probsStart = b * T * V + t * V;
      const ix = targets[b * T + t];
      losses[b * T + t] = -Math.log(probs[probsStart+ix]);
    }
  }
}

proc crossentropy_softmax_backward(ref dlogits, dlogitsOff,
                                   const ref dlosses, dlossesOff,
                                   const ref probs, probsOff,
                                   const ref targets,
                                   B, T, V) {
  // backwards through both softmax and crossentropy
  for b in 0..<B {
    for t in 0..<T {
      const dlogitsStart = dlogitsOff + b * T * V + t * V;
      const probsStart = probsOff + b * T * V + t * V;
      const dloss = dlosses[dlossesOff + b * T + t];
      const ix = targets[b * T + t];
      for (i, dlogitsIdx, probsIdx) in zip(0..#V, dlogitsStart.., probsStart..) {
        const p = probs[probsIdx];
        const indicator = (if i == ix then 1.0 else 0.0): real(32);
        dlogits[dlogitsIdx] += (p - indicator) * dloss;
      }
    }
  }
}
inline proc matmul_backward(ref dinp, dinpOff,
                            ref dweight, dweightOff,
                            /*ref dbias, dbiasOff,*/
                            const ref dout, doutOff,
                            const ref inp, inpOff,
                            ref weight, weightOff,
                            B, T, C, OC) {
  var bias = [0.0:real(32)];
  matmul_backward_help(dinp, dinpOff,
                       dweight, dweightOff,
                       bias, 0,
                       dout, doutOff,
                       inp, inpOff,
                       weight, weightOff,
                       B, T, C, OC,
                       hasBias=false);
}

inline proc matmul_backward(ref dinp, dinpOff,
                            ref dweight, dweightOff,
                            ref dbias, dbiasOff,
                            const ref dout, doutOff,
                            const ref inp, inpOff,
                            ref weight, weightOff,
                            B, T, C, OC) {

  matmul_backward_help(dinp, dinpOff,
                       dweight, dweightOff,
                       dbias, dbiasOff,
                       dout, doutOff,
                       inp, inpOff,
                       weight, weightOff,
                       B, T, C, OC,
                       hasBias=true);
}

proc matmul_backward_help(ref dinp, dinpOff,
                          ref dweight, dweightOff,
                          ref dbias, dbiasOff,
                          const ref dout, doutOff,
                          const ref inp, inpOff,
                          ref weight, weightOff,
                          B, T, C, OC,
                          hasBias: bool) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
  forall (b,t) in {0..<B, 0..<T} {
    const doutStart = doutOff + b * T * OC + t * OC;
    const dinpStart = dinpOff + b * T * C + t * C;
    for (o, doutIdx) in zip(0..<OC, doutStart..) {
      const weightStart = weightOff + o*C;
      const d = dout[doutIdx];
      for (dinpIdx, weightIdx) in zip(dinpStart..#C, weightStart..) {
        dinp[dinpIdx] += weight[weightIdx] * d;
      }
    }
  }
    // backward into weight/bias, parallelize over output channels OC
  forall o in 0..<OC {
    for b in 0..<B {
      for t in 0..<T {
        const doutStart = doutOff + b * T * OC + t * OC;
        const inpStart = inpOff + b * T * C + t * C;
        const dweightStart = dweightOff + o*C;
        const d = dout[doutStart + o];
        if hasBias then dbias[o] += d;
        for (dweightIdx, inpIdx) in zip(dweightStart..#C, inpStart..) {
          dweight[dweightIdx] += inp[inpIdx] * d;
        }
      }
    }
  }
}

proc layernorm_backward(ref dinp, dinpOff,
                        ref dweight, dweightOff,
                        ref dbias, dbiasOff,
                        const ref dout, doutOff,
                        const ref inp, inpOff,
                        const ref weight, weightOff,
                        const ref mean, meanOff,
                        const ref rstd, rstdOff,
                        B, T, C) {
  for b in 0..<B {
    for t in 0..<T {
      const doutStart = doutOff + b * T * C + t * C;
      const inpStart = inpOff + b * T * C + t * C;
      const dinpStart = dinpOff + b * T * C + t * C;
      const mean_bt = mean[meanOff + b * T + t];
      const rstd_bt = rstd[rstdOff + b * T + t];

      // first: two reduce operations
      var dnorm_mean = 0.0:real(32);
      var dnorm_norm_mean = 0.0:real(32);
      for (inpIdx, weightIdx, doutIdx,) in zip(inpStart..#C, weightOff..,
                                               doutStart..)  {
        const norm_bti = (inp[inpIdx] - mean_bt) * rstd_bt;
        const dnorm_i = weight[weightIdx] * dout[doutIdx];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
      }
      dnorm_mean = dnorm_mean / C;
      dnorm_norm_mean = dnorm_norm_mean / C;

      // now iterate again and accumulate all the gradients
      for (inpIdx, weightIdx, doutIdx, dbiasIdx, dweightIdx, dinpIdx) in
          zip(inpStart..#C, weightOff.., doutStart.., dbiasOff.., dweightOff..,
              dinpStart..) {
        const norm_bti = (inp[inpIdx] - mean_bt) * rstd_bt;
        const dnorm_i = weight[weightIdx] * dout[doutIdx];
        // gradient contribution to bias
        dbias[dbiasIdx] += dout[doutIdx];
        // gradient contribution to weight
        dweight[dweightIdx] += norm_bti * dout[doutIdx];
        // gradient contribution to input
        var dval = 0.0: real(32);
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp[dinpIdx] += dval;
      }
    }
  }
}

proc residual_backward(ref dinp1, dinp1Off,
                       ref dinp2, dinp2Off,
                       const ref dout, doutOff,
                       N) {
  for (dinp1Idx, dinp2Idx, doutIdx) in zip(dinp1Off..#N, dinp2Off..,
                                           doutOff..) {
    dinp1[dinp1Idx] += dout[doutIdx];
    dinp2[dinp2Idx] += dout[doutIdx];
  }
}

proc gelu_backward(ref dinp, dinpOff,
                   const ref inp, inpOff,
                   const ref dout, doutOff,
                   N) {
  const s = Math.sqrt(2.0:real(32) / Math.pi);
  for (inpIdx, dinpIdx, doutIdx) in zip(inpOff..#N, dinpOff.., doutOff..) {
    const x = inp[inpIdx];
    const cube = 0.044715:real(32) * x * x * x;
    const tanh_arg = s * (x + cube);
    const tanh_out = Math.tanh(tanh_arg);
    const coshf_out = Math.cosh(tanh_arg);
    const sech_out = 1.0:real(32) / (coshf_out * coshf_out);
    const local_grad = (0.5 * (1.0 + tanh_out) +
                       x * 0.5 * sech_out * s *
                       (1.0 + 3.0 * 0.044715 * x * x)): real(32);
    dinp[dinpIdx] += local_grad * dout[doutIdx];
  }
}

proc attention_backward(ref dinp, dinpOff,
                        ref dpreatt, dpreattOff,
                        ref datt, dattOff,
                        const ref dout, doutOff,
                        const ref inp, inpOff,
                        const ref att, attOff,
                        B, T, C, NH) {
  // inp/dinp are (B, T, 3C) Q,K,V
  // att/datt/dpreatt are (B, NH, T, T)
  // dout is (B, T, C)
  const C3 = C*3;
  const hs = C / NH; // head size
  const scale = 1.0 / Math.sqrt(hs);

  for b in 0..<B {
    for t in 0..<T {
      for h in 0..<NH {
        const attStart = attOff + b*NH*T*T + h*T*T + t*T;
        const dattStart = dattOff + b*NH*T*T + h*T*T + t*T;
        const dpreattStart = dpreattOff + b*NH*T*T + h*T*T + t*T;
        const dinpStart = dinpOff + b * T * C3 + t * C3 + h * hs;
        const inpStart = inpOff + b * T * C3 + t * C3 + h * hs;

        // backward pass 4, through the value accumulation
        const doutStart = doutOff + b * T * C + t * C + h * hs;
        for (t2, dattIdx, attIdx) in zip(0..t, dattStart.., attStart..) {
          const innerInpStart = inpOff + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
          const innerDinpStart = dinpOff + b * T * C3 + t2 * C3 + h * hs + C*2;
          for (inpIdx, dinpIdx, doutIdx) in zip(innerInpStart..#hs,
                                                innerDinpStart..,
                                                doutStart..) {
            // in the forward pass this was:
            // out_bth[i] += att_bth[t2] * value_t2[i];
            // so now we have:
            datt[dattIdx] += inp[inpIdx] * dout[doutIdx];
            dinp[dinpIdx] += att[attIdx] * dout[doutIdx];
          }
        }

        // backward pass 2 & 3, the softmax
        // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
        for (t2, attIdxT2, dattIdx) in zip(0..t, attStart.., dattStart..) {
          for (t3, attIdxT3, dpreattIdx) in zip(0..t, attStart.., dpreattStart..) {
            const indicator = (if t2 == t3 then 1.0 else 0.0):real(32);
            const local_derivative = att[attIdxT2] * (indicator - att[attIdxT3]);
            dpreatt[dpreattIdx] += local_derivative * datt[dattIdx];
          }
        }

        // backward pass 1, the query @ key matmul
        for (t2, dpreattIdx) in zip(0..t, dpreattStart..) {
          const innerInpStart = inpOff + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
          const innerDinpStart = dinpOff + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
          for (innerInpIdx, innerDinpIdx, inpIdx, dinpIdx) in
              zip(innerInpStart..#hs, innerDinpStart.., inpStart..,
                  dinpStart..) {
            // in the forward pass this was:
            // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
            // so now we have:
            dinp[dinpIdx] += inp[innerInpIdx] * dpreatt[dpreattIdx] * scale;
            dinp[innerDinpIdx] += inp[inpIdx] * dpreatt[dpreattIdx] * scale;
          }
        }
      }
    }
  }
}

proc encoder_backward(ref dwte, dwteOff,
                      ref dwpe, dwpeOff,
                      const ref dout, doutOff,
                      const ref inp, inpOff,
                      B, T, C) {
  for b in 0..<B {
    for t in 0..<T {
      const doutStart = doutOff + b * T * C + t * C;
      const ix = inp[inpOff + b * T + t];
      const dwteStart = dwteOff + ix * C;
      const dwpeStart = dwpeOff + t * C;
      for (doutIdx, dwteIdx, dwpeIdx) in zip(doutStart..#C, dwteStart..,
                                             dwpeStart..) {
        const d = dout[doutIdx];
        dwte[dwteIdx] += d;
        dwpe[dwpeIdx] += d;
      }
    }
  }
}

class ActivationTensors {
  const B, T, C, L, NH, V: int;
  param nonArrayArgs = 6+1; // 6 is what we have above, 1 is self

  var encoded: [0..#(B*T*C)] real(32);
  var ln1: [0..#(L*B*T*C)] real(32);
  var ln1_mean: [0..#(L*B*T)] real(32);
  var ln1_rstd: [0..#(L*B*T)] real(32);
  var qkv: [0..#(L*B*T*3*C)] real(32);
  var atty: [0..#(L*B*T*C)] real(32);
  var preatt: [0..#(L*B*NH*T*T)] real(32);
  var att: [0..#(L*B*NH*T*T)] real(32);
  var attproj: [0..#(L*B*T*C)] real(32);
  var residual2: [0..#(L*B*T*C)] real(32);
  var ln2: [0..#(L*B*T*C)] real(32);
  var ln2_mean: [0..#(L*B*T)] real(32);
  var ln2_rstd: [0..#(L*B*T)] real(32);
  var fch: [0..#(L*B*T*4*C)] real(32);
  var fch_gelu: [0..#(L*B*T*4*C)] real(32);
  var fcproj: [0..#(L*B*T*C)] real(32);
  var residual3: [0..#(L*B*T*C)] real(32);
  var lnf: [0..#(B*T*C)] real(32);
  var lnf_mean: [0..#(B*T)] real(32);
  var lnf_rstd: [0..#(B*T)] real(32);
  var logits: [0..#(B*T*V)] real(32);
  var probs: [0..#(B*T*V)] real(32);
  var losses: [0..#(B*T)] real(32);

  proc totalSize() {
    var sum: int;
    for param i in nonArrayArgs..#NUM_ACTIVATION_TENSORS {
      ref f = Reflection.getFieldRef(this, i);
      compilerAssert(isArray(f), "Expected a field to be an array ",
                                 "did you change the struct?");
      sum += f.size;
    }
    return sum;
  }
}

// could consider making this a helper, or put in a class hierarchy?
proc ActivationTensors.zeroAll() {
  for param i in nonArrayArgs..#NUM_ACTIVATION_TENSORS {
    ref f = Reflection.getFieldRef(this, i);
    compilerAssert(isArray(f), "Expected a field to be an array ",
                               "did you change the struct?");
    f = 0;
  }
}


record GPT2 {
    var gpt_config: GPT2Config;
    // the weights of the model
    var params: owned ParameterTensors?;
    var num_parameters: int(32);
    // gradients of the weights
    var grads: owned ParameterTensors?;
    // buffers for the AdamW optimizer
    var m_memory: [0..#num_parameters] real(32);
    var v_memory: [0..#num_parameters] real(32);
    // the activations of the model, and their sizes
    var acts: owned ActivationTensors?;
    var num_activations: int(32);
    // gradients of the activations
    var grads_acts: owned ActivationTensors?;
    // other run state configuration
    var batch_size: int(32); // the batch size (B) of current forward pass
    var seq_len: int(32); // the sequence length (T) of current forward pass
    var inputsDom = {0..#0}; // ENGIN:it is unclear to me whether we might need
                            // to resize the array associated with this domain
    var inputs: [inputsDom] int(32); // the input tokens for the current forward pass
    var targetsDom = {0..#0}; // ENGIN:it is unclear to me whether we might need
                             // to resize the array associated with this domain
    var targets: [targetsDom] int(32); // the target tokens for the current forward pass
    var mean_loss: real(32); // after a forward pass with targets, will be populated with the mean loss
}

proc GPT2.init(checkpoint_path, B, T) {

  // TODO prefer try/catch here
  var model_file = try! open(checkpoint_path, ioMode.r);

  var model_header: [0..#256] int(32);

  const reader = model_file.reader(locking=false);
  reader.readBinary(model_header);

  if model_header[0] != 20240326 { halt("Bad magic model file"); }
  if model_header[1] != 1 { halt("Bad version in model file"); }

  var maxT, V, L, NH, C: int(32);

  maxT = model_header[2];
  V = model_header[3];
  L = model_header[4];
  NH = model_header[5];
  C = model_header[6];

  writef("[GPT-2]\n");
  writef("max_seq_len: %i\n", maxT);
  writef("vocab_size: %i\n", V);
  writef("num_layers: %i\n", L);
  writef("num_heads: %i\n", NH);
  writef("channels: %i\n", C);

  this.params = new ParameterTensors(V, C, maxT, L);
  this.num_parameters = this.params!.numTotalParams():int(32);

  this.inputsDom = {0..#(B*T)};
  this.targetsDom = {0..#(B*T)};

  this.mean_loss = -1.0; // -1.0f will designate no loss

  init this;

  this.params!.readFrom(reader);

  this.gpt_config.max_seq_len = maxT;
  this.gpt_config.vocab_size = V;
  this.gpt_config.num_layers = L;
  this.gpt_config.num_heads = NH;
  this.gpt_config.channels = C;

  writef("num_parameters: %i\n", this.num_parameters);
}

proc ref GPT2.forward(ref loader: DataLoader, B, T) {
  ref inputs = loader.batch[loader.inputsDom];
  // TODO this is a mess because of #12178
  ref targets = loader.batch[loader.targetsDom].reindex(loader.inputsDom);

  this.forward(inputs, targets, B, T);
}

proc ref GPT2.forward(inputs: [], B, T) {
  this.forward(inputs, [min(int(32))], B, T);
}

proc ref GPT2.forward(inputs: [] int(32), targets: [] int(32), B, T) {

  if this.params == nil {
    halt("Error: model was not initialized properly.");
  }

  const haveTargets = !(targets.size == 1 && targets[0] == min(int(32)));

  // convenience parameters
  const V = this.gpt_config.vocab_size;
  const L = this.gpt_config.num_layers;
  const NH = this.gpt_config.num_heads;
  const C = this.gpt_config.channels;

  if this.acts == nil {
    this.batch_size = B;
    this.seq_len = T;

    this.acts = new ActivationTensors(B, T, C, L, NH, V);
    this.num_activations = this.acts!.totalSize():int(32);
    writef("num_activations: %i\n", num_activations);
  }
  else {
    if B > this.batch_size || T > this.seq_len {
      writef("Error: batch size or sequence length is inadequately large\n");
      writef("Model: B=%i T=%i, Desired: B=%i T=%i\n", this.batch_size, this.seq_len, B, T);
      halt();
    }
  }

  // cache the inputs/target
  this.inputs = inputs;
  if haveTargets then
    this.targets = targets;

  // forward pass
  const ref params = this.params!; // for brevity
  const ref acts = this.acts!;  // strip away nilability while here

  encoder_forward(acts.encoded, this.inputs, params.wte, params.wpe, B, T, C);
  for l in 0..<L {
    // TODO for l!=0, this could be a direct reindex where we can negative shift
    // the indices
    ref _residual = if l==0 then acts.encoded[acts.encoded.domain]
                            else acts.residual3[((l-1)*B*T*C)..];
    ref residual = _residual.reindex(0..#_residual.size);

    // get the ~pointers~ offsets of the weights for this layer
    const l_ln1w = l * C;
    const l_ln1b = l * C;
    const l_qkvw = l * 3*C * C;
    const l_qkvb = l * 3*C;
    const l_attprojw = l * C * C;
    const l_attprojb = l * C;
    const l_ln2w = l * C;
    const l_ln2b = l * C;
    const l_fcw = l * 4*C * C;
    const l_fcb = l * 4*C;
    const l_fcprojw = l * C * 4*C;
    const l_fcprojb = l * C;

    // get the ~pointers~ of the activations for this layer
    const l_ln1 = l * B * T * C;
    const l_ln1_mean = l * B * T;
    const l_ln1_rstd = l * B * T;
    const l_qkv = l * B * T * 3*C;
    const l_atty = l * B * T * C;
    const l_preatt = l * B * NH * T * T;
    const l_att = l * B * NH * T * T;
    const l_attproj = l * B * T * C;
    const l_residual2 = l * B * T * C;
    const l_ln2 = l * B * T * C;
    const l_ln2_mean = l * B * T;
    const l_ln2_rstd = l * B * T;
    const l_fch = l * B * T * 4*C;
    const l_fch_gelu = l * B * T * 4*C;
    const l_fcproj = l * B * T * C;
    const l_residual3 = l * B * T * C;

    layernorm_forward(acts.ln1, l_ln1,
                      acts.ln1_mean, l_ln1_mean,
                      acts.ln1_rstd, l_ln1_rstd,
                      residual, 0,
                      params.ln1w, l_ln1w,
                      params.ln1b, l_ln1b,
                      B, T, C);

    matmul_forward(acts.qkv, l_qkv,
                   acts.ln1, l_ln1,
                   params.qkvw, l_qkvw,
                   params.qkvb, l_qkvb,
                   B, T, C, 3*C);

    attention_forward(acts.atty, l_atty,
                      acts.preatt, l_preatt,
                      acts.att, l_att,
                      acts.qkv, l_qkv,
                      B, T, C, NH);

    matmul_forward(acts.attproj, l_attproj,
                   acts.atty, l_atty,
                   params.attprojw, l_attprojw,
                   params.attprojb, l_attprojb,
                   B, T, C, C);

    residual_forward(acts.residual2, l_residual2,
                     residual, 0,
                     acts.attproj, l_attproj,
                     B*T*C);

    layernorm_forward(acts.ln2, l_ln2,
                      acts.ln2_mean, l_ln2_mean,
                      acts.ln2_rstd, l_ln2_rstd,
                      acts.residual2, l_residual2,
                      params.ln2w, l_ln2w,
                      params.ln2b, l_ln2b,
                      B, T, C);

    matmul_forward(acts.fch, l_fch,
                   acts.ln2, l_ln2,
                   params.fcw, l_fcw,
                   params.fcb, l_fcb,
                   B, T, C, 4*C);

    gelu_forward(acts.fch_gelu, l_fch_gelu,
                 acts.fch, l_fch,
                 B*T*4*C);

    matmul_forward(acts.fcproj, l_fcproj,
                   acts.fch_gelu, l_fch_gelu,
                   params.fcprojw, l_fcprojw,
                   params.fcprojb, l_fcprojb,
                   B, T, 4*C, C);

    residual_forward(acts.residual3, l_residual3,
                     acts.residual2, l_residual2,
                     acts.fcproj, l_fcproj,
                     B*T*C);
  }
  const residualOff = (L-1) * B * T * C;
  layernorm_forward(acts.lnf, 0,
                    acts.lnf_mean, 0,
                    acts.lnf_rstd, 0,
                    acts.residual3, residualOff, // last residual is in residual3
                    params.lnfw, 0,
                    params.lnfb, 0,
                    B, T, C);

  matmul_forward(acts.logits, 0,
                 acts.lnf, 0,
                 params.wte, 0,
                 B, T, C, V);

  softmax_forward(acts.probs, 0,
                  acts.logits, 0,
                  B, T, V);

  if haveTargets {
    crossentropy_forward(acts.losses, acts.probs, targets, B, T, V);
    // for convenience also evaluate the mean loss
    var mean_loss = 0.0: real(32);
    for i in 0..<B*T { mean_loss += acts.losses[i]; } // TODO reduce?

    writeln("calculated mean_loss ", mean_loss);
    mean_loss /= B*T;
    this.mean_loss = mean_loss;
  }
  else {
    this.mean_loss = -1.0;
  }
}

proc GPT2.zeroGrad() {
  if this.grads != nil then this.grads!.zeroAll();
  if this.grads_acts != nil then this.grads_acts!.zeroAll();
}

proc ref GPT2.backward() {
  // double check we forwarded previously, with targets
  if this.mean_loss == -1.0 {
    halt("Error: must forward with targets before backward");
  }

  // convenience shortcuts
  const B = this.batch_size;
  const T = this.seq_len;
  const maxT = this.gpt_config.max_seq_len;
  const V = this.gpt_config.vocab_size;
  const L = this.gpt_config.num_layers;
  const NH = this.gpt_config.num_heads;
  const C = this.gpt_config.channels;

  // lazily allocate the memory for gradients of the weights and activations, if needed
  if this.grads == nil {
    this.grads = new ParameterTensors(V, C, maxT, L);
    this.grads_acts = new ActivationTensors(B, T, C, L, NH, V);
    this.zeroGrad();
  }

  // we kick off the chain by filling in dlosses with 1.0f/(B*T), to get the mean loss
  const dloss_mean = (1.0 / (B*T)): real(32);
  for i in 0..<(B*T) {
    grads_acts!.losses[i] = dloss_mean;
  }
  crossentropy_softmax_backward(grads_acts!.logits, 0,
                                grads_acts!.losses, 0,
                                acts!.probs, 0,
                                targets,
                                B, T, V);
  matmul_backward(grads_acts!.lnf, 0,
                  grads!.wte, 0,
                  grads_acts!.logits, 0,
                  acts!.lnf, 0,
                  params!.wte, 0,
                  B, T, C, V);

  layernorm_backward(grads_acts!.residual3, (L-1) * B * T * C,
                     grads!.lnfw, 0,
                     grads!.lnfb, 0,
                     grads_acts!.lnf, 0,
                     acts!.residual3, (L-1) * B * T * C,
                     params!.lnfw, 0,
                     acts!.lnf_mean, 0,
                     acts!.lnf_rstd, 0,
                     B, T, C);

  for l in 0..L-1 by -1 {
    // TODO for l!=0, this could be a direct reindex where we can negative shift
    // the indices
    ref _residual = if l==0 then acts!.encoded[acts!.encoded.domain]
                            else acts!.residual3[((l-1)*B*T*C)..];
    ref residual = _residual.reindex(0..#_residual.size);

    ref _dresidual = if l == 0 then grads_acts!.encoded[grads_acts!.encoded.domain]
                               else grads_acts!.residual3[(l-1) * B * T * C..];
    ref dresidual = _dresidual.reindex(0..#_dresidual.size);

    // get the pointers of the weights for this layer
    const l_ln1w = l * C;
    const l_qkvw = l * 3*C * C;
    const l_attprojw = l * C * C;
    const l_ln2w = l * C;
    const l_fcw = l * 4*C * C;
    const l_fcprojw = l * C * 4*C;
    // get the pointers of the gradients of the weights for this layer
    const dl_ln1w = l * C;
    const dl_ln1b = l * C;
    const dl_qkvw = l * 3*C * C;
    const dl_qkvb = l * 3*C;
    const dl_attprojw = l * C * C;
    const dl_attprojb = l * C;
    const dl_ln2w = l * C;
    const dl_ln2b = l * C;
    const dl_fcw = l * 4*C * C;
    const dl_fcb = l * 4*C;
    const dl_fcprojw = l * C * 4*C;
    const dl_fcprojb = l * C;
    // get the pointers of the activations for this layer
    const l_ln1 = l * B * T * C;
    const l_ln1_mean = l * B * T;
    const l_ln1_rstd = l * B * T;
    const l_qkv = l * B * T * 3*C;
    const l_atty = l * B * T * C;
    const l_att = l * B * NH * T * T;
    const l_residual2 = l * B * T * C;
    const l_ln2 = l * B * T * C;
    const l_ln2_mean = l * B * T;
    const l_ln2_rstd = l * B * T;
    const l_fch = l * B * T * 4*C;
    const l_fch_gelu = l * B * T * 4*C;
    // get the pointers of the gradients of the activations for this layer
    const dl_ln1 = l * B * T * C;
    const dl_qkv = l * B * T * 3*C;
    const dl_atty = l * B * T * C;
    const dl_preatt = l * B * NH * T * T;
    const dl_att = l * B * NH * T * T;
    const dl_attproj = l * B * T * C;
    const dl_residual2 = l * B * T * C;
    const dl_ln2 = l * B * T * C;
    const dl_fch = l * B * T * 4*C;
    const dl_fch_gelu = l * B * T * 4*C;
    const dl_fcproj = l * B * T * C;
    const dl_residual3 = l * B * T * C;

    residual_backward(grads_acts!.residual2, dl_residual2,
                      grads_acts!.fcproj, dl_fcproj,
                      grads_acts!.residual3, dl_residual3,
                      B*T*C);

    matmul_backward(grads_acts!.fch_gelu, dl_fch_gelu,
                    grads!.fcprojw, dl_fcprojw,
                    grads!.fcprojb, dl_fcprojb,
                    grads_acts!.fcproj, dl_fcproj,
                    acts!.fch_gelu, l_fch_gelu,
                    params!.fcprojw, l_fcprojw,
                    B, T, 4*C, C);

    gelu_backward(grads_acts!.fch, dl_fch,
                  acts!.fch, l_fch,
                  grads_acts!.fch_gelu, dl_fch_gelu,
                  B*T*4*C);

    matmul_backward(grads_acts!.ln2, dl_ln2,
                    grads!.fcw, dl_fcw,
                    grads!.fcb, dl_fcb,
                    grads_acts!.fch, dl_fch,
                    acts!.ln2, l_ln2,
                    params!.fcw, l_fcw,
                    B, T, C, 4*C);

    layernorm_backward(grads_acts!.residual2, dl_residual2,
                       grads!.ln2w, dl_ln2w,
                       grads!.ln2b, dl_ln2b,
                       grads_acts!.ln2, dl_ln2,
                       acts!.residual2, l_residual2,
                       params!.ln2w, l_ln2w,
                       acts!.ln2_mean, l_ln2_mean,
                       acts!.ln2_rstd, l_ln2_rstd,
                       B, T, C);

    residual_backward(dresidual, 0,
                      grads_acts!.attproj, dl_attproj,
                      grads_acts!.residual2, dl_residual2,
                      B*T*C);

    matmul_backward(grads_acts!.atty, dl_atty,
                    grads!.attprojw, dl_attprojw,
                    grads!.attprojb, dl_attprojb,
                    grads_acts!.attproj, dl_attproj,
                    acts!.atty, l_atty,
                    params!.attprojw, l_attprojw,
                    B, T, C, C);

    attention_backward(grads_acts!.qkv, dl_qkv,
                       grads_acts!.preatt, dl_preatt,
                       grads_acts!.att, dl_att,
                       grads_acts!.atty, dl_atty,
                       acts!.qkv, l_qkv,
                       acts!.att, l_att,
                       B, T, C, NH);

    matmul_backward(grads_acts!.ln1, dl_ln1,
                    grads!.qkvw, dl_qkvw,
                    grads!.qkvb, dl_qkvb,
                    grads_acts!.qkv, dl_qkv,
                    acts!.ln1, l_ln1,
                    params!.qkvw, l_qkvw,
                    B, T, C, 3*C);

    layernorm_backward(dresidual, 0,
                       grads!.ln1w, dl_ln1w,
                       grads!.ln1b, dl_ln1b,
                       grads_acts!.ln1, dl_ln1,
                       residual, 0,
                       params!.ln1w, l_ln1w,
                       acts!.ln1_mean, l_ln1_mean,
                       acts!.ln1_rstd, l_ln1_rstd,
                       B, T, C);

  }

  encoder_backward(grads!.wte, 0,
                   grads!.wpe, 0,
                   grads_acts!.encoded, 0,
                   inputs, 0,
                   B, T, C);
}


proc ref GPT2.update(learning_rate, beta1, beta2, eps, weight_decay, t) {
  // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

  for (cur_param, cur_grad, i) in zip(this.params!, this.grads!,
                                      0..<this.num_parameters) {

    // update the first moment (momentum)
    const m = beta1 * this.m_memory[i] + (1.0 - beta1) * cur_grad;
    // update the second moment (RMSprop)
    const v = beta2 * this.v_memory[i] + (1.0 - beta2) * cur_grad * cur_grad;
    // bias-correct both moments
    const m_hat = m / (1.0 - beta1**t);
    const v_hat = v / (1.0 - beta2**t);

    // update
    this.m_memory[i] = m:real(32);
    this.v_memory[i] = v:real(32);
    cur_param -= (learning_rate * (m_hat / (Math.sqrt(v_hat) + eps) +
                                            weight_decay * cur_param)):real(32);
  }
}

proc random_u32(ref state: uint): uint(32) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return ((state * 0x2545F4914F6CDD1D) >> 32):uint(32);
}

proc random_f32(ref state: uint): real(32) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0:real(32);
}

proc sample_mult(const probabilities,
                 const probabilities_off,
                 const n, const coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    var cdf = 0.0: real(32);
    for (i, probIdx) in zip(0..<n, probabilities_off..) {
        cdf += probabilities[probIdx];
        if coin < cdf {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

proc main() {

  const B = 4:int(32);
  const T = 64:int(32);

  var model = new GPT2("gpt2_124M.bin", B, T);

  // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
  const tiny_stories_train = "data/TinyStories_train.bin";
  const tiny_stories_val = "data/TinyStories_val.bin";
  const tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
  const tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
  const train_tokens = if FileSystem.exists(tiny_shakespeare_train)
                           then tiny_shakespeare_train
                           else tiny_stories_train;

  const val_tokens = if FileSystem.exists(tiny_shakespeare_val)
                           then tiny_shakespeare_val else tiny_stories_val;
  var val_num_batches = 10:int(32);

  var my_train_loader = new DataLoader(train_tokens, B, T);
  writef("train dataset num_batches: %i\n", my_train_loader.num_batches);
  var my_val_loader = new DataLoader(val_tokens, B, T);
  writef("val dataset num_batches: %i\n", my_val_loader.num_batches);



  // some memory for generating samples from the model
  var rng_state = 1337:uint;
  const gen_max_length = 64:int(32);
  var gen_tokens: [0..#gen_max_length] int(32);

  // train
  var t: Time.stopwatch;
  for step in 0..40 {

    // once in a while estimate the validation loss
    if (step % 10 == 0) {
      var val_loss = 0.0:real(32);
      my_val_loader.reset();
      for i in 0..<val_num_batches {
        my_val_loader.nextBatch();
        model.forward(my_val_loader, B, T);
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      writef("val loss %r\n", val_loss);
    }


    // once in a while do model inference to print generated text
    if (step > 0 && step % 5 == 0) {
      gen_tokens[0] = GPT2_EOT; // the GPT-2 EOT token kicks off the generation
      for t in 1..<gen_max_length {
        // note that inference is wasteful here because
        // for each t, we re-compute all activations between 0 and t
        // leaving this alone because you want separate code for inference anyway
        // the inference here is just for sanity checking purposes
        model.forward(gen_tokens, 1:int(32), t);
        var coin = random_f32(rng_state);
        var next_token = sample_mult(model.acts!.probs,
                                     (t-1)*model.gpt_config.vocab_size,
                                     model.gpt_config.vocab_size,
                                     coin);
        gen_tokens[t] = next_token;
      }
      writef("generated: ");
      for t in 0..< gen_max_length {
        writef("%i ", gen_tokens[t]);
      }
      writef("\n");
    }

    // do a training step
    t.start();
    my_train_loader.nextBatch();
    model.forward(my_train_loader, B, T);
    model.backward();
    t.stop();
    writef("step %i: train loss %r (took %r ms)\n",
           step, model.mean_loss, t.elapsed() * 1000);
    t.reset();
  }
}
