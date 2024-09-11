# The Life of a Numba Kernel

An overview of the compiler and runtime pipeline that Numba uses to take Python
source code, turn it into a CUDA kernel, and launch it. It pulls in a variety
of Numba internals to illustrate how the different parts of the pipeline work.

See the notebook [Life of a Numba Kernel with output](Life%20of%20a%20Numba%20Kernel%20-%20with-%20output.ipynb)
