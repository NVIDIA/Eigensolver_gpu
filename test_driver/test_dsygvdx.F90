!
! Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

module funcs
  contains

  ! Creates pseudo-random positive-definite symmetric matrix
  subroutine create_random_symmetric_pd(A, N)
    use cudafor
    use cublas
    real(8), allocatable, dimension(:,:)         :: A, temp
    real(8), allocatable, dimension(:,:), device :: A_d, temp_d
    real(8)                                         :: rv
    integer                                         :: i, j, N

    allocate(A(N,N))
    allocate(temp(N,N))

    ! Create general symmetric temp
    do j = 1, N
      do i = 1, N
        if (i > j) then
          call random_number(rv)
          temp(i,j) = rv
          temp(j,i) = rv
        else if (i == j) then
          call random_number(rv)
          temp(i,j) = rv
        end if
      end do
    end do

    allocate(A_d, source = A)
    allocate(temp_d, source = temp)

    ! Multiply temp by transpose of temp to get positive definite A
    call cublasdgemm('N', 'T', N, N, N, 1.d0, temp_d, N, temp_d, N, 0.d0, A_d, N)

    A = A_d
    deallocate(temp)
    deallocate(A_d)
    deallocate(temp_d)
        
  end subroutine
end module funcs

! cusovlerDnDsygvdx was added in CUDA 10.1
! PGI did not expose CUDA_VERSION before version 19.7. This line can be simplified once the support for PGI 19.4 and earlier has been dropped
#if (((__PGIF90__ < 19 || (__PGIF90__ == 19 && __PGIF90_MINOR__ < 7)) && __CUDA_API_VERSION >= 10010) || ((__PGIF90__ > 19 || (__PGIF90__ == 19 && __PGIF90_MINOR__ >= 7)) && __CUDA_VERSION >= 10010))
#define HAVE_CUSOLVERDNDSYGVDX
#endif

program main
  use cudafor
  use cublas
  use cusolverDn
  use eigsolve_vars, ONLY: init_eigsolve_gpu
  use dsygvdx_gpu
  use nvtx_inters
  use funcs
  use compare_utils
  implicit none
  
  integer                                         :: N, M, i, j, info, lda, istat
  integer                                         :: lwork_d, lrwork_d, lwork, lrwork, liwork, il, iu
#ifdef HAVE_CUSOLVERDNDSYGVDX
  integer                                         :: h_meig
#endif
  character(len=20)                               :: arg
  real(8)                                         :: ts, te, wallclock
  real(8), dimension(:,:), allocatable            :: A1, A2, Aref
  real(8), dimension(:,:), allocatable            :: B1, B2, Bref
  real(8), dimension(:,:), allocatable, pinned    :: Z1, Z2
  real(8), dimension(:,:), allocatable, device    :: A2_d, B2_d, Z2_d
  real(8), dimension(:), allocatable, pinned      :: work
  real(8), dimension(:), allocatable, pinned      :: w1, w2, rwork
  integer, dimension(:), allocatable, pinned      :: iwork
  real(8), dimension(:), allocatable, device      :: work_d
  real(8), dimension(:), allocatable, device      :: w2_d, rwork_d
  integer, device                                 :: devInfo_d
  type(cusolverDnHandle)                          :: h

  ! Parse command line arguments
  i = command_argument_count()

  if (i >= 1) then
    ! If N is provided, generate random symmetric matrices for A and B
    print*, "Using randomly-generated matrices..."
    call get_command_argument(1, arg)
    read(arg, *)  N
    lda = N

    ! Create random positive-definite hermetian matrices on host
    call create_random_symmetric_pd(Aref, N)
    call create_random_symmetric_pd(Bref, N)

  else
    print*, "Usage:\n\t ./main [N]"
    call exit
  endif

  print*, "Running with N = ", N

  ! Allocate/Copy matrices to device
  allocate(A1, source = Aref)
  allocate(A2, source = Aref)
  allocate(A2_d, source = Aref)
  allocate(B1, source = Bref)
  allocate(B2, source = Bref)
  allocate(B2_d, source = Bref)
  allocate(Z1, source = Aref)
  allocate(Z2, source = Aref)
  allocate(Z2_d, source = Aref)

  allocate(w1(N), w2(N))
  allocate(w2_d, source = w2)

  ! Initialize solvers
  call init_eigsolve_gpu()

  istat = cublasInit
  if (istat /= CUBLAS_STATUS_SUCCESS) write(*,*) 'cublas intialization failed'

  istat = cusolverDnCreate(h)
  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'handle creation failed'

#ifdef HAVE_MAGMA
  call magmaf_init
#endif


  !! Solving generalized eigenproblem using DSYGVD
  ! CASE 1: CPU _____________________________________________
  print*
  print*, "CPU_____________________"
  lwork = 1 + 6*N + 2*N*N
  liwork = 3 + 5*N
  allocate(iwork(liwork))
  allocate(work(lwork))
  call dsygvd(1, 'V', 'U', N, A1, lda, B1, lda, w1, work, -1, iwork, -1, istat)
  if (istat /= 0) write(*,*) 'CPU dsygvd worksize failed'
  lwork = work(1);; liwork = iwork(1)
  deallocate(work, iwork )
  allocate(work(lwork), iwork(liwork))

  A1 = Aref
  B1 = Bref
  ! Run once before timing
  call dsygvd(1, 'V', 'U', N, A1, lda, B1, lda, w1, work, lwork, iwork, liwork, istat)
  if (istat /= 0) write(*,*) 'CPU dsygvd failed. istat = ', istat

  A1 = Aref
  B1 = Bref
  ts = wallclock()
  call nvtxStartRange("CPU DSYGVD",1)
  call dsygvd(1, 'V', 'U', N, A1, lda, B1, lda, w1, work, lwork, iwork, liwork, istat)
  call nvtxEndRange
  te = wallclock()
  if (istat /= 0) write(*,*) 'CPU dsygvd failed. istat = ', istat

  print*, "\tTime for CPU dsygvd = ", (te - ts)*1000.0
  print*

#ifdef HAVE_MAGMA
  ! CASE 2: using Magma ___________________________________________
  print*
  print*, "MAGMA_____________________"
  call magmaf_dsygvd(1, 'V', 'U', N, A2, lda, B2, lda, w2, work, -1, iwork, -1, istat)
  if (istat /= 0) write(*,*) 'magmaf_dsygvd buffer sizes failed',istat
  deallocate(work, iwork)
  allocate(work(lwork), iwork(liwork))

  ts = wallclock()
  call nvtxStartRange("MAGMA",0)
  call magmaf_dsygvd(1, 'V', 'U', N, A2, lda, B2, lda, w2, work, lwork, iwork, liwork, istat)
  call nvtxEndRange
  te = wallclock()
  if (istat /= 0) write(*,*) 'magmaf_dsygvd failed',istat

  print*, "evalues/evector accuracy: (compared to CPU results)"
  call compare(w1, w2, N)
  call compare(A1, A2, N, N)
  print*

  print*, "Time for magmaf_dsygvd = ", (te - ts)*1000.0
  print*
#endif


  ! CASE 3: using Cusolver __________________________________________________________________
  !print*
  print*, "cuSOLVER_____________________"

#ifdef HAVE_CUSOLVERDNDSYGVDX
  il = 1
  iu = N
  istat = cusolverDnDsygvdx_bufferSize(h, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER, N, A2_d, lda, B2_d, lda, 0.D0, 0.D0, il, iu, h_meig, w2_d, lwork_d)
  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'cusolverDnDsygvdx_buffersize failed'
#else
  istat = cusolverDnDsygvd_bufferSize(h, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, A2_d, lda, B2_d, lda, w2_d, lwork_d)
  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'cusolverDnDsygvd_buffersize failed'
#endif
  allocate(work_d(lwork_d))
  
  A2 = Aref
  B2 = Bref
  w2 = 0
  A2_d = A2
  B2_d = B2
  w2_d = 0
  ts = wallclock()
  call nvtxStartRange("cuSOLVER",5)
#ifdef HAVE_CUSOLVERDNDSYGVDX
  istat = cusolverDnDsygvdx(h, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER, N, A2_d, lda, B2_d, lda, 0.D0, 0.D0, il, iu, h_meig, w2_d, work_d, lwork_d, devInfo_d)
#else
  istat = cusolverDnDsygvd(h, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, A2_d, lda, B2_d, lda, w2_d, work_d, lwork_d, devInfo_d)
#endif
  call nvtxEndRange
  te = wallclock()

  print*, "evalues/evector accuracy: (compared to CPU results)"
  w2 = w2_d
  A2 = A2_d
  call compare(w1, w2, N)
  call compare(A1, A2, N, N)
  print*

#ifdef HAVE_CUSOLVERDNDSYGVDX
  print*, "Time for cusolverDnDsygvdx = ", (te - ts)*1000.0
#else
  print*, "Time for cusolverDnDsygvd = ", (te - ts)*1000.0
#endif
  print*
  istat = devInfo_d
#ifdef HAVE_CUSOLVERDNDSYGVDX
  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'cusolverDnDsygvdx failed'
#else
  if (istat /= CUSOLVER_STATUS_SUCCESS) write(*,*) 'cusolverDnDsygvd failed'
#endif


  ! CASE 4: using CUSTOM ____________________________________________________________________
  print*
  print*, "CUSTOM_____________________"
  A2 = Aref
  B2 = Bref
  w2 = 0
  A2_d = A2
  B2_d = B2
  w2_d = w2
  il = 1
  iu = N

  deallocate(work, iwork)
  lwork = 1+6*N+2*N*N
  liwork = 3+5*N
  allocate(work(lwork), iwork(liwork))

  deallocate(work_d)
  lwork_d = 2*64*64 + 66 * N
  allocate(work_d(1*lwork_d))

  ts = wallclock()
  call nvtxStartRange("Custom",0)
  call dsygvdx_gpu(N, A2_d, lda, B2_d, lda, Z2_d, lda, il, iu, w2_d, work_d, lwork_d, &
                          work, lwork, iwork, liwork, Z2, lda, w2, istat)
  call nvtxEndRange
  te = wallclock()

  print*, "evalues/evector accuracy: (compared to CPU results)"
  call compare(w1, w2, iu)
  call compare(A1, Z2, N, iu)
  print*

  print*, "Time for CUSTOM dsygvd/x = ", (te - ts)*1000.0
  if (istat /= 0) write(*,*) 'dsygvdx_gpu failed'

end program
