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

module dsyevd_gpu
  use cudafor
  use cublas
  implicit none

 contains

    ! Custom dsyevd routine
    subroutine dsyevd_gpu(jobz, uplo, il, iu, N, A, lda, Z, ldz, w, work, lwork, &
                          work_h, lwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info)
      use dsytrd_gpu
      use eigsolve_vars
      use nvtx_inters
      implicit none
      character                                   :: uplo, jobz
      integer                                     :: N, NZ, lda, lwork, istat, info
      integer                                     :: lwork_h, liwork_h, ldz_h
      integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu
      real(8), dimension(1:lwork), device         :: work
      real(8), dimension(1:lwork_h)               :: work_h
      integer, dimension(1:liwork_h)              :: iwork_h

      real(8), dimension(1:lda, 1:N), device      :: A
      real(8), dimension(1:lda, 1:N), device      :: Z
      real(8), dimension(1:ldz_h, 1:N), pinned    :: Z_h
      real(8), dimension(1:N), device             :: w
      real(8), dimension(1:N), pinned             :: w_h

      integer                                     :: inde, indtau, indwrk, llwork, llwork_h, indwk2, indwk3, llwrk2
      real(8), parameter                          :: one = 1.0_8

      type(dim3) :: blocks, threads


      if (uplo .ne. 'U' .or. jobz .ne. 'V') then
        print*, "Provided itype/uplo not supported!"
        return
      endif

      nb1 = 32 ! Blocksize for tridiagonalization
      nb2 = min(64, N) ! Blocksize for rotation procedure, fixed at 64
      ldt = nb2
      NZ = iu - il + 1

      inde = 1
      indtau = inde + n
      indwrk = indtau + n
      llwork = lwork - indwrk + 1
      llwork_h = lwork_h - indwrk + 1
      indwk2 = indwrk + (nb2)*(nb2)
      indwk3 = indwk2 + (nb2)*(nb2)

      !JR Note: ADD SCALING HERE IF DESIRED. Not scaling for now.

      ! Call DSYTRD to reduce A to tridiagonal form
      call nvtxStartRange("dytrd", 0)
      call dsytrd_gpu('U', N, A, lda, w, work(inde), work(indtau), work(indwrk), llwork, nb1)
      call nvtxEndRange

      ! Copy diagonal and superdiagonal to CPU
      w_h(1:N) = w(1:N)
      work_h(inde:inde+N-1) = work(inde:inde+N-1)

      ! Restore lower triangular of A (works if called from zhegvd only!)
      !$cuf kernel do(2) <<<*,*>>>
      do j = 1,N
        do i = 1,N
          if (i > j) then
            A(i,j) = Z(i,j)
          endif
        end do
      end do

      ! Call DSTEDC to get eigenvalues/vectors of tridiagonal A on CPU
      call nvtxStartRange("dstedc", 1)
      call dstedc('I', N, w_h, work_h(inde), Z_h, ldz_h, work_h(indwrk), llwork_h, iwork_h, liwork_h, istat)
      if (istat /= 0) then
        write(*,*) "dsyevd_gpu error: dstedc failed!"
        info = -1
        return
      endif
      call nvtxEndRange

      ! Copy eigenvectors and eigenvalues to GPU
      istat = cudaMemcpy2D(Z(1, 1), ldz, Z_h, ldz_h, N, NZ)
      w(1:N) = w_h(1:N)

      !! Call DORMTR to rotate eigenvectors to obtain result for original A matrix
      !! JR Note: Eventual function calls from DORMTR called directly here with associated indexing changes
      call nvtxStartRange("dormtr", 2)

      istat = cudaEventRecord(event2, stream2)

      k = N-1

      do i = 1, k, nb2
        ib = min(nb2, k-i+1)

        ! Form block reflector T in stream 1
        call dlarft_gpu(i+ib-1, ib, A(1, 2+i-1), lda, work(indtau + i -1), work(indwrk), ldt, work(indwk2), ldt)

        mi = i + ib - 1
        ! Apply reflector to eigenvectors in stream 2
        call dlarfb_gpu(mi, NZ, ib, A(1,2+i-1), lda, work(indwrk), ldt, Z, ldz, work(indwk3), N, work(indwk2), ldt)
      end do

      call nvtxEndRange

    end subroutine dsyevd_gpu

    subroutine dlarft_gpu(N, K, V, ldv, tau, T, ldt, W, ldw)
      use cublas
      use eigsolve_vars
      implicit none
      integer                               :: N, K, ldv, ldt, ldw
      real(8), dimension(ldv, K), device    :: V
      real(8), dimension(K), device         :: tau
      real(8), dimension(ldt, K), device    :: T
      real(8), dimension(ldw, K), device    :: W

      integer                               :: i, j, istat
      type(dim3)                            :: threads

      istat = cublasSetStream(cuHandle, stream1)

      ! Prepare lower triangular part of block column for dsyrk call. 
      ! Requires zeros in lower triangular portion and ones on diagonal.
      ! Store existing entries (excluding diagonal) in W
      !$cuf kernel do(2) <<<*, *, 0, stream1>>>
      do j = 1, K
        do i = N-K + 1, N
          if (i-N+K == j) then
            V(i, j) = 1.0d0
          else if (i-N+k > j) then
            W(i-N+k,j) = V(i,j)
            V(i,j) = 0.0d0
          endif
        end do
      end do

      istat = cudaEventRecord(event1, stream1)
      istat = cudaStreamWaitEvent(stream1, event2, 0)

      ! Form preliminary T matrix
      istat = cublasdsyrk_v2(cuHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, K, N, 1.0_8, V, ldv, 0.0_8, T, ldt)

      ! Finish forming T 
      threads = dim3(64, 16, 1)
      call finish_T_block_kernel<<<1, threads, 0, stream1>>>(K, T, ldt, tau)

    end subroutine dlarft_gpu

    subroutine dlarfb_gpu(M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork, W, ldw)
      use cublas
      use eigsolve_vars
      implicit none
      integer                               :: M, N, K, ldv, ldt, ldc, ldw, ldwork, istat
      integer                               :: i, j
      real(8), dimension(ldv, K), device    :: V
      real(8), dimension(ldt, K), device    :: T
      real(8), dimension(ldw, K), device    :: W
      real(8), dimension(ldc, N), device    :: C
      real(8), dimension(ldwork, K), device :: work

      istat = cublasSetStream(cuHandle, stream2)

      istat = cudaStreamWaitEvent(stream2, event1, 0)
      istat = cublasdgemm_v2(cuHandle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, M, 1.0d0, C, ldc, v, ldv, 0.0d0, work, ldwork)
      istat = cudaStreamSynchronize(stream1)

      istat = cublasdtrmm_v2(cuHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, N, K, &
        1.0d0, T, ldt, work, ldwork, work, ldwork)

      istat = cudaEventRecord(event2, stream2)
      istat = cublasdgemm_v2(cuHandle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, -1.0d0, V, ldv, work, ldwork, 1.0d0, c, ldc)

      ! Restore clobbered section of block column (except diagonal)
      !$cuf kernel do(2) <<<*, *>>>
      do j = 1, K
        do i = M-K + 1, M
          if (i-M+k > j) then
            V(i,j) = W(i-M+k,j)
          endif
        end do
      end do

    end subroutine dlarfb_gpu

    attributes(global) subroutine finish_T_block_kernel(N, T, ldt, tau)
      implicit none
      integer, value                     :: N, ldt
      real(8), dimension(ldt, K), device :: T
      real(8), dimension(K), device      :: tau
      ! T_s contains only lower triangular elements of T in linear array, by row
      real(8), dimension(2080), shared   :: T_s 
      ! (i,j) --> ((i-1)*i/2 + j)
      #define IJ2TRI(i,j) (ISHFT((i-1)*i,-1) + j)


      integer     :: tid, tx, ty, i, j, k, diag
      complex(8)  :: cv

      tx = threadIdx%x
      ty = threadIdx%y
      tid = (threadIdx%y - 1) * blockDim%x + tx ! Linear thread id

      ! Load T into shared memory
      if (tx <= N) then
        do j = ty, N, blockDim%y
          cv = tau(j)
          if (tx > j) then
            T_s(IJ2TRI(tx,j)) = -cv*T(tx,j)
          else if (tx == j) then
            T_s(IJ2TRI(tx,j)) = cv
          endif
        end do
      end if

      call syncthreads()

      ! Perform column by column update by first thread column
      do i = N-1, 1, -1
        if (ty == 1) then
          if (tx > i .and. tx <= N) then
            cv = 0.0d0
            do j = i+1, tx
                cv = cv + T_s(IJ2TRI(j, i)) * T_s(IJ2TRI(tx, j))
            end do
          endif
          
        endif

        call syncthreads()
        if (ty == 1 .and. tx > i .and. tx <= N) then
          T_s(IJ2TRI(tx, i)) = cv
        endif
        call syncthreads()

      end do

      call syncthreads()


      ! Write T_s to global
      if (tx <= N) then
        do j = ty, N, blockDim%y
          if (tx >= j) then
            T(tx,j) = T_s(IJ2TRI(tx,j))
          endif
        end do
      end if

    end subroutine finish_T_block_kernel

end module dsyevd_gpu
