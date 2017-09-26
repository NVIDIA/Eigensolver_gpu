!
! Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
! 
! 
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.
!

! Module containing various handles used for GPU eigensolver
module eigsolve_vars
  use cudafor
  use cublas
  use cusolverDn
  integer                        :: initialized = 0
  type(cublasHandle)             :: cuHandle
  type(cusolverDnHandle)         :: cusolverHandle
  type(cudaEvent)                :: event1, event2, event3
  integer(kind=cuda_stream_kind) :: stream1, stream2, stream3
  integer, device                :: devInfo_d
  integer, device, allocatable   :: finished(:)

  contains

    subroutine init_eigsolve_gpu()
      use cudafor
      use cublas
      implicit none
      integer istat
      if( initialized == 0 ) then
         ! Configure shared memory to use 8 byte banks
         istat = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) 

         istat = cublasCreate(cuHandle)
         istat = cusolverDnCreate(cusolverHandle)
         istat = cudaStreamCreate(stream1)
         istat = cudaStreamCreate(stream2)
         istat = cudaStreamCreate(stream3)
         istat = cudaEventCreate(event1)
         istat = cudaEventCreate(event2)
         initialized = 1
         allocate(finished(1))
         finished(1) = 0
      endif
    end subroutine init_eigsolve_gpu

end module eigsolve_vars
