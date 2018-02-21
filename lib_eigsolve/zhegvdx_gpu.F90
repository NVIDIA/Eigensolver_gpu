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

module zhegvdx_gpu
  use cudafor
  use cublas
  implicit none

  contains

    ! zhegvdx_gpu
    ! This solver computes eigenvalues and associated eigenvectors over a specified integer range for a
    ! hermetian-definite eigenproblem in the following form:
    !     A * x = lambda * B * x
    ! where A and B are hermetian-matrices and B is positive definite. The solver expects the upper-triangular parts of the
    ! input A and B arguments to be populated. This configuration corresponds to calling ZHEGVX within LAPACK with the configuration
    ! arguments 'ITYPE = 1', 'JOBZ = 'V'', 'RANGE = 'I'', and 'UPLO = 'U''.
    !
    ! Input: 
    ! On device:
    !   -  A(lda, N), B(ldb, N) are double-complex matrices on device  with upper triangular portion populated
    !   -  il, iu are integers specifying range of eigenvalues/vectors to compute. Range is [il, iu]
    !   -  work is a double-complex array for complex workspace of length lwork. 
    !   -  lwork is an integer specifying length of work. lwork >=  2*64*64 + 65*N
    !   -  rwork is a real(8) array for real workspace of length lrwork. 
    !   -  lrwork is an integer specifying length of rwork. lrwork >= N
    !
    ! On host:
    !   -  work_h is a double-complex array for complex workspace of length lwork_h. 
    !   -  lwork_h is an integer specifying length of work_h. lwork_h >= N
    !   -  rwork_h is a real(8) array for complex workspace of length lrwork_h. 
    !   -  lrwork_h is an integer specifying length of rwork_h. lrwork_h >= 1 + 5*N + 2*N*N
    !   -  iwork_h is a integer array for integer workspace of length liwork_h. 
    !   -  liwork_h is an integer specifying length of iwork_h. liwork_h >= 3 + 5*N
    !   -  (optional) _skip_host_copy is an optional logical argument. If .TRUE., memcopy of final updated eigenvectors from 
    !      device to host will be skipped.
    !
    ! Output:
    ! On device:
    !   - A(lda, N), B(ldb, N) are modified on exit. The upper triangular part of A, including the diagonal is destroyed. 
    !     B is overwritten by the triangular Cholesky factor U corresponding to  B = U**H * U
    !   - Z(ldz, N) is a double-complex matrix on the device. On exit, the first iu - il + 1 columns of Z
    !     contains normalized eigenvectors corresponding to eigenvalues in the range [il, iu].
    !   - w(N) is a real(8) array on the device. On exit, the first iu - il + 1 values of w contain the computed
    !     eigenvalues
    !
    ! On host:
    !   - Z_h(ldz_h, N) is a double-complex matrix on the host. On exit, the first iu - il + 1 columns of Z
    !     contains normalized eigenvectors corresponding to eigenvalues in the range [il, iu]. This is a copy of the Z
    !     matrix on the device. This is only true if optional argument _skip_host_copy is not provided or is set to .FALSE.
    !   - w_h(N) is a real(8) array on the host. On exit, the first iu - il + 1 values of w contain the computed
    !     eigenvalues. This is a copy of the w array on the host.
    !   - info is an integer. info will equal zero if the function completes succesfully. Otherwise, there was an error.
    !
    subroutine zhegvdx_gpu(N, A, lda, B, ldb, Z, ldz, il, iu, w, work, lwork, rwork, lrwork, &
                          work_h, lwork_h, rwork_h, lrwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info, _skip_host_copy)
      use eigsolve_vars
      use nvtx_inters
      use zhegst_gpu
      use zheevd_gpu
      implicit none
      integer                                     :: N, m, lda, ldb, ldz, il, iu, ldz_h, info, nb
      integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat
      real(8), dimension(1:lrwork), device        :: rwork
      real(8), dimension(1:lrwork_h), pinned      :: rwork_h
      complex(8), dimension(1:lwork), device      :: work
      complex(8), dimension(1:lwork_h), pinned    :: work_h
      integer, dimension(1:liwork_h), pinned      :: iwork_h
      logical, optional                           :: _skip_host_copy

      complex(8), dimension(1:lda, 1:N), device   :: A
      complex(8), dimension(1:ldb, 1:N), device   :: B
      complex(8), dimension(1:ldz, 1:N), device   :: Z
      complex(8), dimension(1:ldz_h, 1:N), pinned :: Z_h
      real(8), dimension(1:N), device             :: w
      real(8), dimension(1:N), pinned             :: w_h

      complex(8), parameter :: cone = cmplx(1,0,8)
      integer :: i, j
      logical :: skip_host_copy

      info = 0
      skip_host_copy = .FALSE.
      if(present(_skip_host_copy)) skip_host_copy = _skip_host_copy

      ! Check workspace sizes
      if (lwork < 2*64*64 + 65*N) then
        print*, "zhegvdx_gpu error: lwork must be at least 2*64*64 + 65*N"
        info = -1
        return
      else if (lrwork < N) then
        print*, "zhegvdx_gpu error: lrwork must be at least N"
        info = -1
        return
      else if (lwork_h < N) then 
        print*, "zhegvdx_gpu error: lwork_h must be at least N"
        info = -1
        return
      else if (lrwork_h < 1 + 5*N + 2*N*N) then 
        print*, "zhegvdx_gpu error: lrwork_h must be at least 1 + 5*N + 2*N*N"
        info = -1
        return
      else if (liwork_h < N) then 
        print*, "zhegvdx_gpu error: liwork_h must be at least 3 + 5*N"
        info = -1
        return
      endif

      m = iu - il + 1 ! Number of eigenvalues/vectors to compute

      if(initialized == 0) call init_eigsolve_gpu

      ! Compute cholesky factorization of B
      call nvtxStartRange("cusolverdnZpotrf", 0)
      istat = cusolverDnZpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER, N, B, ldb, work, lwork, devInfo_d)
      istat = devInfo_d
      call nvtxEndRange
      if (istat .ne. 0) then
        print*, "zhegvdx_gpu error: cusolverDnZpotrf failed!"
        info = -1
        return
      endif

      ! Store lower triangular part of A in Z
      !$cuf kernel do(2) <<<*,*, 0, stream1>>>
      do j = 1,N
        do i = 1,N
          if (i > j) then
            Z(i,j) = A(i,j)
          endif
        end do
      end do


      ! Reduce to standard eigenproblem
      nb = 448
      call nvtxStartRange("zhegst_gpu", 1)
      call zhegst_gpu(1, 'U', N, A, lda, B, ldb, nb)
      call nvtxEndRange

      ! Tridiagonalize and compute eigenvalues/vectors
      call nvtxStartRange("zheevd_gpu", 2)
      call zheevd_gpu('V', 'U', il, iu, N, A, lda, Z, ldz, w, work, lwork, rwork, lrwork, &
                      work_h, lwork_h, rwork_h, lrwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info)
      call nvtxEndRange

      ! Triangle solve to get eigenvectors for original general eigenproblem
      call nvtxStartRange("cublasZtrsm", 3)
      call cublasZtrsm('L', 'U', 'N', 'N', N, (iu - il + 1), cone, B, ldb, Z, ldz) 
      call nvtxEndRange

      ! Copy final eigenvectors to host
      if (not(skip_host_copy)) then
        istat = cudaMemcpy2D(Z_h, ldz_h, Z, ldz, N, m)
        if (istat .ne. 0) then
          print*, "zhegvdx_gpu error: cudaMemcpy2D failed!"
          info = -1
          return
        endif
      endif

    end subroutine zhegvdx_gpu

end module zhegvdx_gpu
