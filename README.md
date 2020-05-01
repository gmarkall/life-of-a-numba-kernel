# The Life of a Numba Kernel

An overview of the compiler and runtime pipeline that Numba uses to take Python
source code, turn it into a CUDA kernel, and launch it. It pulls in a variety
of Numba internals to illustrate how the different parts of the pipeline work.

Contents:

- [Life of a Numba Kernel](Life of a Numba Kernel.ipynb): The notebook ready to
  execute.
- [Life of a Numba Kernel with output](Life of a Numba Kernel - with- output.ipynb):
  The notebook with output - good for viewing without needing to compile /
  execute, perhaps if you don't have a CUDA device or installation of Numba
  handy.
- [life-of-a-numba-kernel.py](life-of-a-numba-kernel.py): Example code
  executable as a Python script.
