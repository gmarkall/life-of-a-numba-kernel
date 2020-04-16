from numba import cuda
import numpy as np

# Numba CUDA target is three things:

# 1. A NumPy-like array library backed by CUDA
# 2. A Python-to-PTX compiler that uses NVVM
# 3. A Python interface to the CUDA driver API


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

from dis import dis # noqa
dis(axpy.py_func)


# Bytecode interpretation

from numba.core.bytecode import ByteCode, FunctionIdentity # noqa
from numba.core.interpreter import Interpreter             # noqa

fi = FunctionIdentity.from_function(axpy.py_func)
interp = Interpreter(fi)
bc = ByteCode(fi)
ir = interp.interpret(bc)

# Control flow analysis

interp.cfa.dump()
dg = interp.cfa.graph.render_dot()
dg.render()  # or just dg in notebook

# Data flow analysis

interp.dfa.infos


# Numba IR

ir.dump()
ir_dg = ir.render_dot()

# Rewrite passes happen

# Typing occurs

# Typed IR

axpy.inspect_types()

# Typed for one definition
axpy.definitions

# Let's add another definition

x = x.astype(np.float32)
y = y.astype(np.float32)
r = r.astype(np.float32)
axpy[blocks, threads](r, a, x, y)

axpy.definitions

# Rewrites on typed IR happen

# Generate LLVM IR

for ((cc, types), llir) in axpy.inspect_llvm().items():
    print(f'Compute capability {cc} / argtypes {types}:\n')
    print(llir)
    print()

# LLVM to PTX

from numba.core.compiler_lock import global_compiler_lock # noqa
from numba.cuda.cudadrv import nvvm                       # noqa
from numba.cuda.compiler import compile_cuda              # noqa
from numba import float32, int32, void                    # noqa

# Have to cheat a bit here to get everything needed to give to NVVM
with global_compiler_lock:
    argtys = (float32[:], int32, float32[:], float32[:])
    returnty = void
    cres = compile_cuda(axpy.py_func, void, argtys)
    fname = cres.fndesc.llvm_func_name
    lib, kernel = cres.target_context.prepare_cuda_kernel(cres.library, fname,
                                                          cres.signature.args,
                                                          debug=False)
    llvm_module = lib._final_module

    cc = (5, 2)
    arch = nvvm.get_arch_option(*cc)
    llvmir = str(llvm_module)
    ptx = nvvm.llvm_to_ptx(llvmir, opt=3, arch=arch)

print(ptx.decode('utf-8'))


# PTX to module

from numba.cuda.cudadrv.driver import Linker # noqa

linker = Linker()
linker.add_ptx(ptx)
cubin, size = linker.complete()

compile_info = linker.info_log

print(size)
print(compile_info)

import ctypes # noqa

cubinarr = np.ctypeslib.as_array(ctypes.cast(cubin,ctypes.POINTER(ctypes.c_char)), shape=(size,))

with open('axpy.cubin', 'wb') as f:
    f.write(cubinarr)

# file axpy.cubin

# SASS

# come back to this once toolkit downloaded
# cuobjdump -sass axpy.cubin 

# Load module

from numba.cuda.cudadrv.driver import load_module_image # noqa
from numba.core import itanium_mangler                  # noqa

ctx = cuda.get_current_device().get_primary_context()
module = load_module_image(ctx, cubin)

mangled_name = itanium_mangler.prepend_namespace(fname, ns='cudapy')

cufunc = module.get_function(mangled_name)

type(cufunc)


# Launch kernel

# Copy our arrays to the device - normally Numba does this for us
d_r = cuda.to_device(r)
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)

# A couple of helpers from Numba's CUDA driver implementation
from numba.cuda.cudadrv.driver import (device_pointer, is_device_memory,
                                       device_ctypes_pointer)     # noqa
from ctypes import addressof, c_void_p                            # noqa


# Helper function to create all the args for an array structure
def make_array_args(arr):
    args = []
    c_intp = ctypes.c_ssize_t

    meminfo = ctypes.c_void_p(0)
    parent = ctypes.c_void_p(0)
    nitems = c_intp(arr.size)
    itemsize = c_intp(arr.dtype.itemsize)
    data = ctypes.c_void_p(device_pointer(arr))

    args.append(meminfo)
    args.append(parent)
    args.append(nitems)
    args.append(itemsize)
    args.append(data)

    for ax in range(arr.ndim):
        args.append(c_intp(arr.shape[ax]))
    for ax in range(arr.ndim):
        args.append(c_intp(arr.strides[ax]))

    return args

# Create the list of arguments - we compiled for float32[:], int32, float32[:],
# float32[:]
args = []
args += make_array_args(d_r)
args += [ctypes.c_int(13)]  
args += make_array_args(d_x)
args += make_array_args(d_y)

# Make a list of pointers to the arguments
param_vals = []
for arg in args:
    if is_device_memory(arg):
        param_vals.append(addressof(device_ctypes_pointer(arg)))
    else:
        param_vals.append(addressof(arg))

params = (c_void_p * len(param_vals))(*param_vals)


# (see cudadrv.launch_kernel)

# CUresult cuLaunchKernel(CUfunction f,
#                        unsigned int gridDimX,
#                        unsigned int gridDimY,
#                        unsigned int gridDimZ,
#                        unsigned int blockDimX,
#                        unsigned int blockDimY,
#                        unsigned int blockDimZ,
#                        unsigned int sharedMemBytes,
#                        CUstream hStream, void **kernelParams,
#                        void ** extra)

from numba.cuda.cudadrv.driver import driver # noqa

driver.cuLaunchKernel(cufunc.handle,
                      blocks, 1, 1,
                      threads, 1, 1,
                      0,
                      0,
                      params,
                      None)

h_r = d_r.copy_to_host()
reference = x * 13 + y
print(h_r)
print(reference)

# Sanity check
np.testing.assert_allclose(reference, h_r)
