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

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes

proc encoder_forward(ref outp, const ref inp, const ref wte, const ref wpe,
                     B, T, C) {
  for (b,t) in {0..<B, 0..<T} {
    // get the index of the token at inp[b, t]
    const ix = inp[b * T + t]; // TODO
    // add the two vectors and store the result in out[b,t,:]
    for i in 0..#C {
      outp[b, t, i] = wte[ix,i] + wpe[t,i];
    }
  }
}

proc encoder_backward(ref dwte, ref dwpe, const ref dout, const ref inp,
                      B, T, C) {
  for (b,t) in {0..<B, 0..<T} {
    const ix = inp[b * T + t];
    for i in 0..#C {
      const d = dout[b,t,i];
      dwte[ix, i] += d;
      dwpe[t, i] += d;
    }
  }
}

proc layernorm_forward(ref outp, ref mean, ref rstd, const ref inp,
                       const ref weight, const ref bias, B, T, C) {
  const eps = 1e-5: real(32);
  forall (b,t) in {0..<B, 0..<T} {
    const ref inp_bt = inp[b,t,..];
    // calculate the mean
    const m = (+ reduce inp_bt)/C;

    // calculate the variance (without any bias correction)
    const v = (+ reduce [i in inp_bt] (i-m)**2)/C;

    // calculate the rstd
    const s = 1.0:real(32) / Math.sqrt(v + eps);
    for i in 0..<C {
      const n = (s * (inp[b,t,i] - m)); // normalized output
      const o = n * weight[i] + bias[i]; // scale and shift it
      outp[b,t,i] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    mean[b,t] = m;
    rstd[b,t] = s;
  }
}

proc layernorm_backward(ref dinp, ref dweight, ref dbias, const ref dout,
                        const ref inp, const ref weight, const ref mean,
                        const ref rstd, B, T, C) {
  for (b,t) in {0..<B, 0..<T} {
    const mean_bt = mean[b, t];
    const rstd_bt = rstd[b, t];

    // first: two reduce operations
    var dnorm_mean = 0.0:real(32);
    var dnorm_norm_mean = 0.0:real(32);
    for i in 0..<C {
      const norm_bti = (inp[b,t,i] - mean_bt) * rstd_bt;
      const dnorm_i = weight[i] * dout[b,t,i];
      dnorm_mean += dnorm_i;
      dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for i in 0..<C {
      const norm_bti = (inp[b,t,i] - mean_bt) * rstd_bt;
      const dnorm_i = weight[i] * dout[b,t,i];
      // gradient contribution to bias
      dbias[i] += dout[b,t,i];
      // gradient contribution to weight
      dweight[i] += norm_bti * dout[b,t,i];
      // gradient contribution to input
      var dval = 0.0: real(32);
      dval += dnorm_i; // term 1
      dval -= dnorm_mean; // term 2
      dval -= norm_bti * dnorm_norm_mean; // term 3
      dval *= rstd_bt; // final scale
      dinp[b,t,i] += dval;
    }
  }
}

proc matmul_forward(ref outp, const ref inp, const ref weight,
                    const ref bias, B, T, C, OC) {
  // most of the running time is spent here and in matmul_backward
  // OC is short for "output channels"
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  // out will be (B,T,OC)
  forall (b,t) in {0..<B, 0..<T} {
    for o in 0..<OC {
      var val = if bias.type!=nil.type then bias[o] else 0.0:real(32);
      for i in 0..<C {
        val += inp[b,t,i] * weight[o,i];
      }
      outp[b,t,o] = val;
    }
  }
}

proc matmul_backward(ref dinp, ref dweight, ref dbias, const ref dout,
                     const ref inp, ref weight, B, T, C, OC) {
  // most of the running time is spent here and in matmul_forward
  // this backward could be done in a single "round" of loops
  // but that doesn't afford an efficient parallelization strategy

  // backward into inp first, parallelize over B,T
  forall (b,t) in {0..<B, 0..<T} {
    for o in 0..<OC {
      const d = dout[b,t,o];
      for i in 0..#C {
        dinp[b,t,i] += weight[o,i] * d;
      }
    }
  }
  // backward into weight/bias, parallelize over output channels OC
  forall o in 0..<OC {
    for (b,t) in {0..<B, 0..<T} {
      const d = dout[b,t,o];
      if dbias.size!=1 then dbias[o] += d;
      for i in 0..<C {
        dweight[o,i] += inp[b,t,i] * d;
      }
    }
  }
}

proc attention_forward(ref outp, ref preatt, ref att, const ref inp, B, T, C,
                       NH) {
  // input is (B, T, 3C) Q,K,V
  // preatt, att are (B, NH, T, T)
  // output is (B, T, C)
  const C3 = C*3;
  const hs = C / NH; // head size
  const scale = 1.0 / Math.sqrt(hs);

  forall (b, t, h) in {0..<B, 0..<T, 0..<NH} {
    // pass 1: calculate query dot key and maxval
    var maxval = min(real(32));
    for t2 in 0..t {
      // (query_t) dot (key_t2)
      var val = 0.0: real(32);
      for (queryIdx, keyIdx) in zip(h*hs..#hs, h*hs+C..) {
        val += inp[b,t,queryIdx] * inp[b,t2,keyIdx];
      }
      val *= scale;
      if val > maxval {
        maxval = val;
      }

      preatt[b,h,t,t2] = val;
    }

    // pass 2: calculate the exp and keep track of sum
    var expsum = 0.0: real(32);
    for t2 in 0..t {
      const expv = Math.exp(preatt[b,h,t,t2] - maxval);
      expsum += expv;
      att[b,h,t,t2] = expv;
    }
    var expsum_inv = (if expsum==0.0 then 0.0 else 1.0/expsum):real(32);

    // pass 3: normalize to get the softmax
    for t2 in 0..<T {
      if t2 <= t {
        att[b,h,t,t2] *= expsum_inv;
      } else {
        // causal attention mask. not strictly necessary to set to zero here
        // only doing this explicitly for debugging and checking to PyTorch
        att[b,h,t,t2] = 0.0;
      }
    }

    // pass 4: accumulate weighted values into the output of attention
    for i in h*hs..#hs { outp[b,t,i] = 0.0; }
    for t2 in 0..t {
      const att_btht2 = att[b,h,t,t2];
      // +C*2 because it's value
      for (outpIdx, inpIdx) in zip(h*hs..#hs, h*hs+C*2..) {
        outp[b,t,outpIdx] += att_btht2 * inp[b,t2,inpIdx];
      }
    }
  }
}

proc attention_backward(ref dinp, ref dpreatt, ref datt, const ref dout,
                        const ref inp, const ref att, B, T, C, NH) {
  // inp/dinp are (B, T, 3C) Q,K,V
  // att/datt/dpreatt are (B, NH, T, T)
  // dout is (B, T, C)
  const C3 = C*3;
  const hs = C / NH; // head size
  const scale = 1.0 / Math.sqrt(hs);

  for (b,t,h) in {0..<B, 0..<T, 0..<NH} { // TODO forall?
    // backward pass 4, through the value accumulation
    for t2 in 0..t {
      // +C*2 because it's value
      for (inOff, outOff) in zip((h*hs+C*2)..#hs, (h*hs)..) {
        // in the forward pass this was:
        // out_bth[i] += att_bth[t2] * value_t2[i];
        // so now we have:
        datt[b,h,t,t2] += inp[b,t2,inOff] * dout[b,t,outOff];
        dinp[b,t2,inOff] += att[b,h,t,t2] * dout[b,t,outOff];
      }
    }

    // backward pass 2 & 3, the softmax
    // note that softmax (like e.g. tanh) doesn't need the input (preatt) to
    // backward
    for (t2, t3) in {0..t, 0..t} {
      const indicator = (if t2 == t3 then 1.0 else 0.0):real(32);
      const local_derivative = att[b,h,t,t2] * (indicator - att[b,h,t,t3]);
      dpreatt[b,h,t,t3] += local_derivative * datt[b,h,t,t2];
    }

    // backward pass 1, the query @ key matmul
    for t2 in 0..t {
      // +C because it 's key
      for (i, hhsOff, hhsCOff) in zip(0..<hs, (h*hs).., (h*hs+C)..) {
        // in the forward pass this was:
        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
        // so now we have:
        dinp[b,t2,hhsOff] += inp[b,t2,hhsCOff] * dpreatt[b,h,t,t2] * scale;
        dinp[b,t2,hhsCOff] += inp[b,t2,hhsOff] * dpreatt[b,h,t,t2] * scale;
      }
    }
  }
}

proc gelu_forward(ref outp, const ref inp) {
  const s = Math.sqrt(2.0:real(32) / Math.pi);
  for (x, o) in zip(inp, outp) {
    const cube = 0.044715:real(32) * x * x * x;
    o = (0.5 * x * (1.0 + Math.tanh(s * (x + cube)))):real(32);
  }
}

proc gelu_backward(ref dinp, const ref inp, const ref dout) {
  const s = Math.sqrt(2.0:real(32) / Math.pi);
  forall (di, x, o) in zip(dinp, inp, dout) {
    const cube = 0.044715:real(32) * x * x * x;
    const tanh_arg = s * (x + cube);
    const tanh_out = Math.tanh(tanh_arg);
    const coshf_out = Math.cosh(tanh_arg);
    const sech_out = 1.0:real(32) / (coshf_out * coshf_out);
    const local_grad = (0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * s *
                       (1.0 + 3.0 * 0.044715 * x * x)): real(32);
    di += local_grad * o;
  }
}

proc residual_forward(ref outp, const ref inp1, const ref inp2) {
  outp = inp1 + inp2;
}

proc residual_backward(ref dinp1, ref dinp2, const ref dout) {
  dinp1 += dout;
  dinp2 += dout;
}

proc softmax_forward(ref probs, const ref logits, B, T, V) {
    // output: probs are (B,T,V) of the probabilities
    // input: logits is (B,T,V) of the unnormalized log probabilities
  forall (b,t) in {0..<B, 0..<T} {
    // probs <- softmax(logits)
    const maxval = max reduce logits[b,t,..];

    // Engin: I wanted to write a promoted statement for this, but didn't work.
    // Couldn't reproduce it outside of this context either.
    for (p,l) in zip(probs[b,t,..], logits[b,t,..]) {
      p = Math.exp(l-maxval);
    }

    probs[b,t,..] /= (+ reduce probs[b,t,..]);
  }
}

proc crossentropy_forward(ref losses, const ref probs, const ref targets, B, T,
                          V) {
  // output: losses is (B,T) of the individual losses at each position
  // input: probs are (B,T,V) of the probabilities
  // input: targets is (B,T) of integers giving the correct index in logits
  for (b,t) in {0..<B, 0..<T} {
    // loss = -log(probs[target])
    const ix = targets[b * T + t];  // TODO
    losses[b,t] = -Math.log(probs[b,t,ix]);
  }
}

proc crossentropy_softmax_backward(ref dlogits, const ref dlosses,
                                   const ref probs, const ref targets,
                                   B, T, V) {
  // backwards through both softmax and crossentropy
  for (b,t) in {0..<B, 0..<T} {
    const dloss = dlosses[b, t];
    const ix = targets[b * T + t]; // TODO
    for i in 0..#V {
      const p = probs[b,t,i];
      const indicator = (if i == ix then 1.0 else 0.0): real(32);
      dlogits[b,t,i] += (p - indicator) * dloss;
    }
  }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

param NUM_PARAMETER_TENSORS = 16;
class ParameterTensors {
  const V, C, maxT, L: int;
  param nonArrayArgs = 4+1; // 4 is what we have above, 1 is self

  var wte:      [0..#V, 0..#C] real(32);
  var wpe:      [0..#maxT, 0..#C] real(32);
  var ln1w:     [0..#L, 0..#C] real(32);
  var ln1b:     [0..#L, 0..#C] real(32);
  var qkvw:     [0..#L, 0..#(3*C), 0..#C] real(32);
  var qkvb:     [0..#L, 0..#(3*C)] real(32);
  var attprojw: [0..#L, 0..#C, 0..#C] real(32);
  var attprojb: [0..#L, 0..#C] real(32);
  var ln2w:     [0..#L, 0..#C] real(32);
  var ln2b:     [0..#L, 0..#C] real(32);
  var fcw:      [0..#L, 0..#(4*C), 0..#C] real(32);
  var fcb:      [0..#L, 0..#(4*C)] real(32);
  var fcprojw:  [0..#L, 0..#C, 0..#(4*C)] real(32);
  var fcprojb:  [0..#L, 0..#C] real(32);
  var lnfw:     [0..#C] real(32);
  var lnfb:     [0..#C] real(32);
}

param NUM_ACTIVATION_TENSORS = 23;
class ActivationTensors {
  const B, T, C, L, NH, V: int;
  param nonArrayArgs = 6+1; // 6 is what we have above, 1 is self

  var encoded:   [0..#B, 0..#T, 0..#C] real(32);
  var ln1:       [0..#L, 0..#B, 0..#T, 0..#C] real(32);
  var ln1_mean:  [0..#L, 0..#B, 0..#T] real(32);
  var ln1_rstd:  [0..#L, 0..#B, 0..#T] real(32);
  var qkv:       [0..#L, 0..#B, 0..#T, 0..#(3*C)] real(32);
  var atty:      [0..#L, 0..#B, 0..#T, 0..#C] real(32);
  var preatt:    [0..#L, 0..#B, 0..#NH, 0..#T, 0..#T] real(32);
  var att:       [0..#L, 0..#B, 0..#NH, 0..#T, 0..#T] real(32);
  var attproj:   [0..#L, 0..#B, 0..#T, 0..#C] real(32);
  var residual2: [0..#L, 0..#B, 0..#T, 0..#C] real(32);
  var ln2:       [0..#L, 0..#B, 0..#T, 0..#C] real(32);
  var ln2_mean:  [0..#L, 0..#B, 0..#T] real(32);
  var ln2_rstd:  [0..#L, 0..#B, 0..#T] real(32);
  var fch:       [0..#L, 0..#B, 0..#T, 0..#(4*C)] real(32);
  var fch_gelu:  [0..#L, 0..#B, 0..#T, 0..#(4*C)] real(32);
  var fcproj:    [0..#L, 0..#B, 0..#T, 0..#C] real(32);
  var residual3: [0..#L, 0..#B, 0..#T, 0..#C] real(32);
  var lnf:       [0..#B, 0..#T, 0..#C] real(32);
  var lnf_mean:  [0..#B, 0..#T] real(32);
  var lnf_rstd:  [0..#B, 0..#T] real(32);
  var logits:    [0..#B, 0..#T, 0..#V] real(32);
  var probs:     [0..#B, 0..#T, 0..#V] real(32);
  var losses:    [0..#B, 0..#T] real(32);
}

record GPT2Config {
  var max_seq_len: int(32); // max sequence length, e.g. 1024
  var vocab_size: int(32); // vocab size, e.g. 50257
  var num_layers: int(32); // number of layers, e.g. 12
  var num_heads: int(32); // number of heads in attention, e.g. 12
  var channels: int(32); // number of channels, e.g. 768
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

  // convenience shortcuts
  inline proc B { return this.batch_size; }
  inline proc T { return this.seq_len; }
  inline proc maxT { return this.gpt_config.max_seq_len; }
  inline proc V { return this.gpt_config.vocab_size; }
  inline proc L { return this.gpt_config.num_layers; }
  inline proc NH { return this.gpt_config.num_heads; }
  inline proc C { return this.gpt_config.channels; }
}

proc GPT2.init(checkpoint_path, B, T) {

  var model_file = try! open(checkpoint_path, ioMode.r);
  var model_header: [0..#256] int(32);
  const reader = model_file.reader(locking=false);
  reader.readBinary(model_header);
  if model_header[0] != 20240326 { halt("Bad magic model file"); }
  if model_header[1] != 1 { halt("Bad version in model file"); }

  // read in hyperparameters
  var maxT = model_header[2];
  var V = model_header[3];
  var L = model_header[4];
  var NH = model_header[5];
  var C = model_header[6];
  writef("[GPT-2]\n");
  writef("max_seq_len: %i\n", maxT);
  writef("vocab_size: %i\n", V);
  writef("num_layers: %i\n", L);
  writef("num_heads: %i\n", NH);
  writef("channels: %i\n", C);

  // allocate space for all the parameters and read them in
  this.params = new ParameterTensors(V, C, maxT, L);
  this.num_parameters = this.params!.totalSize():int(32);
  writef("num_parameters: %i\n", this.num_parameters);

  this.inputsDom = {0..#(B*T)};
  this.targetsDom = {0..#(B*T)};

  this.mean_loss = -1.0; // -1.0f will designate no loss

  init this;

  this.gpt_config.max_seq_len = maxT;
  this.gpt_config.vocab_size = V;
  this.gpt_config.num_layers = L;
  this.gpt_config.num_heads = NH;
  this.gpt_config.channels = C;

  this.params!.readFrom(reader);
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

proc ref GPT2.forward(inputs: [] int(32), targets: [] int(32),
                      B=this.B, T=this.T) {
  if this.params == nil {
    halt("Error: model was not initialized properly.");
  }

  const haveTargets = !(targets.size == 1 && targets[0] == min(int(32)));

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

  // do the first layer separately as the residual doesn't exist yet
  forwardLayer(acts, params, B, T, C, NH, l=0, residual=acts.encoded);
  for l in 1..<L {
    forwardLayer(acts, params, B, T, C, NH, l=l,
                 residual=acts.residual3[l-1, .., .., ..]);
  }
  const ref residual = acts.residual3[L-1, .., .., ..]; // last residual is in residual3
  layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
  matmul_forward(acts.logits, acts.lnf, params.wte, nil, B, T, C, V);
  softmax_forward(acts.probs, acts.logits, B, T, V);

  if haveTargets {
    crossentropy_forward(acts.losses, acts.probs, targets, B, T, V);
    // for convenience also evaluate the mean loss
    this.mean_loss = (+ reduce acts.losses)/acts.losses.size;
  }
  else {
    this.mean_loss = -1.0;
  }
}

proc ref GPT2.forwardLayer(const ref acts, const ref params, B, T, C, NH, l,
                           residual) {
  // get the ~pointers~ offsets of the weights for this layer
  ref l_ln1w = params.ln1w[l, ..];
  ref l_ln1b = params.ln1b[l, ..];
  ref l_qkvw = params.qkvw[l, .., ..];
  ref l_qkvb = params.qkvb[l, ..];
  ref l_attprojw = params.attprojw[l, .., ..];
  ref l_attprojb = params.attprojb[l, ..];
  ref l_ln2w = params.ln2w[l, ..];
  ref l_ln2b = params.ln2b[l, ..];
  ref l_fcw = params.fcw[l, .., ..];
  ref l_fcb = params.fcb[l, ..];
  ref l_fcprojw = params.fcprojw[l, .., ..];
  ref l_fcprojb = params.fcprojb[l, ..];

  // get the ~pointers~ of the activations for this layer
  ref l_ln1 = acts.ln1[l, .., .., ..];
  ref l_ln1_mean = acts.ln1_mean[l, .., ..];
  ref l_ln1_rstd = acts.ln1_rstd[l, .., ..];
  ref l_qkv = acts.qkv[l, .., .., ..];
  ref l_atty = acts.atty[l, .., .., ..];
  ref l_preatt = acts.preatt[l, .., .., .., ..];
  ref l_att = acts.att[l, .., .., .., ..];
  ref l_attproj = acts.attproj[l, .., .., ..];
  ref l_residual2 = acts.residual2[l, .., .., ..];
  ref l_ln2 = acts.ln2[l, .., .., ..];
  ref l_ln2_mean = acts.ln2_mean[l, .., ..];
  ref l_ln2_rstd = acts.ln2_rstd[l, .., ..];
  ref l_fch = acts.fch[l, .., .., ..];
  ref l_fch_gelu = acts.fch_gelu[l, .., .., ..];
  ref l_fcproj = acts.fcproj[l, .., .., ..];
  ref l_residual3 = acts.residual3[l, .., .., ..];

  layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
  matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
  attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
  matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
  residual_forward(l_residual2, residual, l_attproj);
  layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
  matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
  gelu_forward(l_fch_gelu, l_fch);
  matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
  residual_forward(l_residual3, l_residual2, l_fcproj);
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

  // lazily allocate the memory for gradients of the weights and activations, if needed
  if this.grads == nil {
    this.grads = new ParameterTensors(V, C, maxT, L);
    this.grads_acts = new ActivationTensors(B, T, C, L, NH, V);
    this.zeroGrad();
  }
  // TODO: I can't pass both nil and an array to dbias of matmul_backward
  // generically. I want dbias to be non-const as if it was an array, I want to
  // modify it. If it is nil it will not be modified anyways, but nil can't be
  // passed to `ref`
  var dummyArr = [0:real(32)];

  // we kick off the chain by filling in dlosses with 1.0f/(B*T), to get the mean loss
  grads_acts!.losses = (1.0 / (B*T)): real(32);

  crossentropy_softmax_backward(grads_acts!.logits, grads_acts!.losses, acts!.probs, targets, B, T, V);
  matmul_backward(grads_acts!.lnf, grads!.wte, dummyArr, grads_acts!.logits, acts!.lnf, params!.wte, B, T, C, V);
  layernorm_backward(grads_acts!.residual3[L-1,..,..,..], grads!.lnfw, grads!.lnfb, grads_acts!.lnf, acts!.residual3[L-1,..,..,..], params!.lnfw, acts!.lnf_mean, acts!.lnf_rstd, B, T, C);

  backwardLayer(residual=acts!.encoded, dresidual=grads_acts!.encoded, l=0);
  for l in 1..L-1 by -1 {
    backwardLayer(residual=acts!.residual3[l-1, .., .., ..],
                  dresidual=grads_acts!.residual3[l-1, .., .., ..],
                  l=l);
  }

  encoder_backward(grads!.wte, grads!.wpe, grads_acts!.encoded, inputs, B, T, C);
}

proc ref GPT2.backwardLayer(ref residual, ref dresidual, l) {
  // get the pointers of the weights for this layer
  ref l_ln1w = params!.ln1w[l, ..];
  ref l_qkvw = params!.qkvw[l, .., ..];
  ref l_attprojw = params!.attprojw[l, .., ..];
  ref l_ln2w = params!.ln2w[l, ..];
  ref l_fcw = params!.fcw[l, .., ..];
  ref l_fcprojw = params!.fcprojw[l, .., ..];
  // get the pointers of the gradients of the weights for this layer
  ref dl_ln1w = grads!.ln1w[l, ..];
  ref dl_ln1b = grads!.ln1b[l, ..];
  ref dl_qkvw = grads!.qkvw[l, .., ..];
  ref dl_qkvb = grads!.qkvb[l, ..];
  ref dl_attprojw = grads!.attprojw[l, .., ..];
  ref dl_attprojb = grads!.attprojb[l, ..];
  ref dl_ln2w = grads!.ln2w[l, ..];
  ref dl_ln2b = grads!.ln2b[l, ..];
  ref dl_fcw = grads!.fcw[l, .., ..];
  ref dl_fcb = grads!.fcb[l, ..];
  ref dl_fcprojw = grads!.fcprojw[l, .., ..];
  ref dl_fcprojb = grads!.fcprojb[l, ..];
  // get the pointers of the activations for this layer
  ref l_ln1 = acts!.ln1[l, .., .., ..];
  ref l_ln1_mean = acts!.ln1_mean[l, .., ..];
  ref l_ln1_rstd = acts!.ln1_rstd[l, .., ..];
  ref l_qkv = acts!.qkv[l, .., .., ..];
  ref l_atty = acts!.atty[l, .., .., ..];
  ref l_att = acts!.att[l, .., .., .., ..];
  ref l_residual2 = acts!.residual2[l, .., .., ..];
  ref l_ln2 = acts!.ln2[l, .., .., ..];
  ref l_ln2_mean = acts!.ln2_mean[l, .., ..];
  ref l_ln2_rstd = acts!.ln2_rstd[l, .., ..];
  ref l_fch = acts!.fch[l, .., .., ..];
  ref l_fch_gelu = acts!.fch_gelu[l, .., .., ..];
  // get the pointers of the gradients of the activations for this layer
  ref dl_ln1 = grads_acts!.ln1[l, .., .., ..];
  ref dl_qkv = grads_acts!.qkv[l, .., .., ..];
  ref dl_atty = grads_acts!.atty[l, .., .., ..];
  ref dl_preatt = grads_acts!.preatt[l, .., .., .., ..];
  ref dl_att = grads_acts!.att[l, .., .., .., ..];
  ref dl_attproj = grads_acts!.attproj[l, .., .., ..];
  ref dl_residual2 = grads_acts!.residual2[l, .., .., ..];
  ref dl_ln2 = grads_acts!.ln2[l, .., .., ..];
  ref dl_fch = grads_acts!.fch[l, .., .., ..];
  ref dl_fch_gelu = grads_acts!.fch_gelu[l, .., .., ..];
  ref dl_fcproj = grads_acts!.fcproj[l, .., .., ..];
  ref dl_residual3 = grads_acts!.residual3[l, .., .., ..];

  residual_backward(dl_residual2, dl_fcproj, dl_residual3);
  matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
  gelu_backward(dl_fch, l_fch, dl_fch_gelu);
  matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
  layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
  residual_backward(dresidual, dl_attproj, dl_residual2);
  matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
  attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
  matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
  layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
}

proc ref GPT2.update(learning_rate, beta1, beta2, eps, weight_decay, t) {
  // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

  // TODO this is parallelizable, but requires a custom iterator
  for (cur_param, cur_grad, i) in zip(this.params!, this.grads!, 0..) {
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

proc ref DataLoader.reset() {
  this.current_position = 0;
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

// ----------------------------------------------------------------------------
// sampler

param GPT2_EOT = 50256:int(32);

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

proc sample_mult(const probabilities, const n, const coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  var cdf = 0.0: real(32);
  for i in  0..<n {
    cdf += probabilities[i];
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
  var my_train_loader = new DataLoader(train_tokens, B, T);
  writef("train dataset num_batches: %i\n", my_train_loader.num_batches);
  var my_val_loader = new DataLoader(val_tokens, B, T);
  writef("val dataset num_batches: %i\n", my_val_loader.num_batches);
  var val_num_batches = 10:int(32);

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
    if (step > 0 && step % 20 == 0) {
      gen_tokens[0] = GPT2_EOT; // the GPT-2 EOT token kicks off the generation
      for t in 1..<gen_max_length {
        // note that inference is wasteful here because
        // for each t, we re-compute all activations between 0 and t
        // leaving this alone because you want separate code for inference anyway
        // the inference here is just for sanity checking purposes
        model.forward(gen_tokens, B=1:int(32), T=t);
        const ref probs = model.acts!.probs[0, t-1, ..];
        var coin = random_f32(rng_state);
        var next_token = sample_mult(probs, model.gpt_config.vocab_size, coin);
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
    model.zeroGrad();
    model.backward();
    model.update(1e-4, 0.9:real(32), 0.999:real(32), 1e-8, 0.0:real(32),
                 step+1);
    t.stop();
    writef("step %i: train loss %r (took %r ms)\n", step, model.mean_loss,
           t.elapsed() * 1000);
    t.reset();
  }
}

proc totalSizeHelper(store, param nonArrayArgs, param arrayArgs) {
  var sum: int;
  for param i in nonArrayArgs..#arrayArgs {
    ref f = Reflection.getFieldRef(store, i);
    compilerAssert(isArray(f), "Expected a field to be an array");
    sum += f.size;
  }
  return sum;
}

inline proc ActivationTensors.totalSize() {
  return totalSizeHelper(this, nonArrayArgs, NUM_ACTIVATION_TENSORS);
}

inline proc ParameterTensors.totalSize() {
  return totalSizeHelper(this, nonArrayArgs, NUM_PARAMETER_TENSORS);
}

proc zeroAllHelper(store, param nonArrayArgs, param arrayArgs) {
  var sum: int;
  for param i in nonArrayArgs..#arrayArgs {
    ref f = Reflection.getFieldRef(store, i);
    compilerAssert(isArray(f), "Expected a field to be an array");
    f = 0;
  }
}

inline proc ActivationTensors.zeroAll() {
  zeroAllHelper(this, nonArrayArgs, NUM_ACTIVATION_TENSORS);
}

proc ParameterTensors.zeroAll() {
  zeroAllHelper(this, nonArrayArgs, NUM_PARAMETER_TENSORS);
}

iter ParameterTensors.these() ref {
  for param i in nonArrayArgs..#NUM_PARAMETER_TENSORS {
    ref f = Reflection.getFieldRef(this, i);
    compilerAssert(isArray(f), "Expected a field to be an array");
    for item in f do yield item;
  }
}

proc ParameterTensors.readFrom(reader) {
  for param i in nonArrayArgs..#NUM_PARAMETER_TENSORS {
    ref f = Reflection.getFieldRef(this, i);
    compilerAssert(isArray(f), "Expected a field to be an array");
    reader.readBinary(f);
  }
}
