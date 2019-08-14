### Eigensolver_gpu
GPU Eigensolver for Quantum ESPRESSO package

###
This library implements a generalized eigensolver for symmetric/hermetian-definite eigenproblems with functionality similar to
the DSYGVD/X or ZHEGVD/X functions available within LAPACK/MAGMA. This solver has less dependencies on CPU computation 
than comparable implementations within MAGMA, which may be of benefit to systems with limited CPU resources or to 
users without access to high-performing CPU LAPACK libraries. 

###
This implementation can be considered as a "proof of concept" and has been written to target the Quantum ESPRESSO
code. As such, this implementation is built only to handle one problem configuration of DSYGVD/X or ZHEGVD/X. Specifically, this
solver computes eigenvalues and associated eigenvectors over a specified integer range for a 
symmetric/hermetian-definite eigenproblem in the following form: 

	A * x = lambda * B * x

where `A` and `B` are symmetric/hermetian-matrices and `B` is positive definite. The solver expects the upper-triangular parts of the 
input `A` and `B` arguments to be populated. This configuration corresponds to calling DSYGVX/ZHEGVX within LAPACK with the configuration 
arguments `ITYPE = 1`, `JOBZ = 'V'`, `RANGE = 'I'`, and `UPLO = 'U'`. 

See comments within `dsygvdx_gpu.F90` or `zhegvdx_gpu.F90` for specific details on usage.

For additional information about the solver with some performance results, see presentation at the following link: (will be added
once available publically on the GTC On-Demand website)

### Building
* Compilation of this library requires the PGI compiler version 18.10 or higher.
* Using the provided `Makefile` will generate a static library object `lib_eigsolve.a` which can included in your
target application. 
* Library requires linking to cuBLAS and cuSOLVER. Use `-Mcuda=cublas,cusolver` flag when linking your application to do this.
* This library also requires linking to a CPU LAPACK library with an implementation of the `zstedc` function.
* If NVTX is enabled with `-DUSE_NVTX` flag, also must link to NVTX. Use `-L${CUDAROOT}/lib64 -lnvToolsExt` flag when linking your application to do this
  where `${CUDAROOT}` is the root directory of your CUDA installation.

An example of using this solver in a program can be found in the `test_driver` subdirectory. This program does a little performance testing
and validation against existing functionality in a linked CPU LAPACK library, cuSOLVER, and MAGMA (if available). 

### License
This code is released under an MIT license which can be found in `LICENSE`. 
