## Chapel port of [llm.c](https://github.com/karpathy/llm.c).

This is an LLM implementation in [Chapel](chapel-lang.org) based on llm.c
Specifically, this is based on a relatively [earlier
commit](https://github.com/karpathy/llm.c/tree/8386e5393c61ec2faf706f3040e68127c2f08398)
of llm.c.

While the base version is relatively succinct, it still relies on pointer
arithmetic and explicit memory management in C. This version uses Chapel's
multidimensional array support and classes to reduce code complexity
significantly without big hits on performance.

We are still in the early stages of adding standalone GPU kernels.

## Quick start

This repo contains all the helper files as the original version, so you don't
need to clone the original. Refer to [that repo's quick
start](https://github.com/karpathy/llm.c/tree/8386e5393c61ec2faf706f3040e68127c2f08398?tab=readme-ov-file#quick-start)
instructions to generate input files.

Very briefly:

```
python3 prepro_tinyshakespeare.py
python3 train_gpt2.py
```

will create the input files.

```
make train_gpt2
```

will compile the application, and

```
./train_gpt2
```

will launch it.


Tested with Chapel version 2.2.0 pre-release (c18eea7692). This code relied on
fixes on 2.1. As such, it is not expected to work with 2.1 or before.

## Contributing

We are looking for contributors! There are two main items that can improve this
implementation:

1. Bring the port up-to-speed with the current upstream version
2. Implement more GPU kernels

If you have other ideas or notice problems, please create an issue. If you
intend to work on any of the existing issues, please drop a comment expressing
your interest to avoid duplicate work.
