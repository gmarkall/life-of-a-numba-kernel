from numba import cuda
import numpy as np

# A simple kernel

@cuda.jit
def axpy(r, a, x, y):
    i = cuda.grid(1)

    if i < len(r):
        r[i] = a * x[i] + y[i]

# Define some data

N = 10000
a = 5.0
x = np.arange(N).astype(np.int32)
y = np.random.randint(100, size=N).astype(np.int32)
r = np.zeros_like(x)

# Launch the kernel
threads = 256
blocks = (N // threads) + 1
axpy[blocks, threads](r, a, x, y)

# Sanity check
assert(np.all(r == a * x + y))

# Python bytecode


