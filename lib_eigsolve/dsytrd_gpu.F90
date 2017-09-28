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

module dsytrd_gpu
  use cudafor
  use cublas

  contains
  
    subroutine dsytrd_gpu(uplo, N, A, lda, d, e, tau, work, lwork, nb)
      use eigsolve_vars
      use dsytd2_gpu
      implicit none
      character                                 :: uplo
      integer                                   :: N, lda, lwork, nb, nx, ldwork, istat
      integer                                   :: i, j, k, kk
      real(8), dimension(1:N), device           :: d
      real(8), dimension(1:N-1), device         :: e
      real(8), dimension(1:lwork), device       :: work
      real(8), dimension(1:lda, 1:N), device    :: A
      real(8), dimension(1:N-1), device         :: tau
      real(8), parameter                        :: one = 1.0_8
      type(dim3)                                :: threads

      if (uplo .ne. 'U') then
        print*, "Provided uplo type not supported!"
        return
      endif

      if (lwork < (nb+2)*N .and. N > nb) then
        write(*,*), "Provided work array must be sized (nb+2)*N or greater!"
        return
      endif

      ldwork = N

      istat = cublasSetStream(cuHandle, stream1)

      kk = N-((N-32) / nb) * nb
      k = N+1
      do i = N-nb+1, kk+1, -nb
        ! Reduce columns i:i+nb-1 to tridiagonal form 
        call dlatrd_gpu(uplo, i+nb-1, nb, A, lda, e, tau, work, ldwork)

        ! Update trailing submatrix
        call cublasdsyr2k(uplo, 'N', i-1, nb, -one, A(1, i), lda, work, ldwork, one, a, lda)

        k = k - nb

      end do
      
      ! Finish any remaining columns to get final 32x32 block
      nb = k - 32 - 1
      i = k - nb
      
      if (nb > 0) then
        ! Reduce columns i:i+nb-1 to tridiagonal form 
        call dlatrd_gpu(uplo, i+nb-1, nb, A, lda, e, tau, work, ldwork)

        ! Update trailing submatrix
        call cublasdsyr2k(uplo, 'N', i-1, nb, -one, A(1, i), lda, work, ldwork, one, a, lda)
      endif

      ! Final block
      threads = dim3(32, 32, 1)
      call dsytd2_gpu<<<1, threads>>>(min(32, N), A, lda, d, e, tau)

      ! Copy superdiagonal back into A, store diagonal in d
      !$cuf kernel do(1) <<<*,*>>>
      do j = 33, N
        !A(j-1, j) = e(j-1) ! JR Not strictly needed so skipping this copy
        d(j) = A(j,j)
      end do

    end subroutine dsytrd_gpu


    subroutine dlatrd_gpu(uplo, N, nb, A, lda, e, tau, W, ldw)
      use eigsolve_vars
      use dsymv_gpu
      implicit none
      character                                  :: uplo
      integer                                    :: N, nb, lda, ldw, istat
      integer                                    :: i, j, k, iw
      integer                                    :: blocks, threads
      real(8), dimension(1:lda, 1:N), device     :: A
      real(8), dimension(1:ldw, 1:nb), device    :: W
      real(8), dimension(N-1), device            :: tau
      real(8), dimension(N-1), device            :: e
      real(8), parameter                         :: one = 1.0d0, zero = 0.0d0, half = 0.5d0

      type(dim3)                                 :: threads2D, blocks2D

      if (uplo .ne. 'U') then
        print*, "Provided uplo type not supported!"
        return
      endif

      threads2D = dim3(32,8,1)
      threads = 256

      if (N <= 0) return

      ! Complete first iteration outside loop
      if (N > 1) then
        iw = nb
        ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)
        call dlarfg_kernel<<<1, threads>>>(N-1, e(N-1), A(1, N), tau(N-1))

        !$cuf kernel do(1) <<<*,*>>>
        do k = 1, N-1
          W(k,iw) = 0.d0
        end do

        blocks2D = dim3(10, ceiling(real(N-1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
        call dsymv_gpu<<<blocks2D, threads2D>>>(N-1, A, lda, A(1, N), W(1, iw))

        call finish_W_col_kernel<<<1, threads>>>(N-1, tau(N-1), A(1, N), W(1, iw))
      endif

      do i = N-1, N-nb+1, -1
        iw = i-N+nb

        blocks2D = dim3(ceiling(real(max(i, N-i))/32), ceiling(real(N-i)/8), 1)
        !call dsyr2_mv_kernel<<<blocks2D, threads2D>>>(i, N-i, A(1, i+1), lda, W(1, iw+1), ldw, A(1, i), W(1, iw), ldw)
        call dsyr2_mv_dlarfg_kernel<<<blocks2D, threads2D>>>(i, N-i, A(1, i+1), lda, W(1, iw+1), ldw, A(1, i), W(1, iw), ldw, e(i-1), tau(i-1), finished(1))

        if (i > 1) then
          ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)
          !call dlarfg_kernel<<<1, threads>>>(i-1, e(i-1), A(1, i), tau(i-1))

          blocks2D = dim3(10, ceiling(real(i-1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
          call dsymv_gpu<<<blocks2D, threads2D>>>(i-1, A, lda, A(1, i), W(1, iw))

          blocks2D = dim3(ceiling(real(i-1)/32), ceiling(real(2*(n-i))/8), 1)
          call stacked_dgemv_T<<<blocks2D, threads2D>>>(n-i, i-1, A(1,i+1), lda, W(1, iw+1), ldw, A(1,i), W(i+1, iw), W(i+1, iw+1))
          !call stacked_dgemv_N<<<blocks2D, threads2D>>>(i-1, n-i, A(1,i+1), lda, W(1, iw+1), ldw, W(i+1,iw), W(i+1, iw+1), W(1, iw))
          call stacked_dgemv_N_finish_W<<<blocks2D, threads2D>>>(i-1, n-i, A(1,i+1), lda, W(1, iw+1), ldw, W(i+1,iw), W(i+1, iw+1), W(1, iw), tau(i-1), A(1, i), finished(1))

          !call finish_W_col_kernel<<<1, threads>>>(i-1, tau(i-1), A(1, i), W(1, iw))

        end if
      end do
    end subroutine dlatrd_gpu

    attributes(global) subroutine dsyr2_mv_kernel(N, M, V, ldv, W, ldw, x, W2, ldw2)
      implicit none
      integer, value                                      :: N, M, ldv, ldw, ldw2
      real(8), dimension(1:ldv, 1:M), device, intent(in)  :: V
      real(8), dimension(1:ldw, 1:M), device, intent(in)  :: W
      real(8), dimension(1:ldw2, 2), device               :: W2
      real(8), dimension(1:N), device                     :: x

      integer                                             :: i, j, istat
      real(8)                                             :: rv

      i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
      j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

      if (i <= N .and. j <= M) then

        rv = -W(N, j) * V(i,j) - V(N, j) * W(i,j)

        ! Update x
        istat = atomicadd(x(i), rv)
      endif

      if (threadIdx%y == 1) then
        ! Zero out column for zhemv call
        if (i <= N) W2(i, 1) = 0
        ! Zero out workspace for intermediate zgemv results
        if (i <= M) then
          W2(N + i, 1) = 0
          W2(N + i, 2) = 0
        endif
      endif

    end subroutine dsyr2_mv_kernel

    attributes(global) subroutine dlarfg_kernel(N, e, x, tau)
      implicit none
      integer, value                   :: N
      real(8), device                  :: tau
      real(8), device                  :: e
      real(8), dimension(N), device    :: x

      integer                          :: tid, i, j, nb, istat, laneID
      real(8)                          :: rv1, rv2, rv3, scal, scal2, alphar, beta, rsum
      real(8), shared                  :: xnorm
      real(8), shared                  :: alpha_s

      tid = threadIdx%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alpha_s = x(N)
        xnorm = 0.0_8
      endif

      call syncthreads()

      alphar = alpha_s
      rsum = 0.0_8

      nb = ceiling(real(N)/blockDim%x) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= N-1) then
          rv1 = x(i)
          rv1 = rv1 * rv1
        else
          rv1 = 0.0_8
        endif

        rsum = rsum + rv1

        i = i + blockDim%x
      end do

      ! Partial sum within warps using shuffle
      rv1 = rsum
      rv2 = __shfl_down(rv1,1)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,2)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,4)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,8)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,16)
      rv1 = rv1 + rv2

      if (laneID == 1) then
        istat = atomicadd(xnorm, rv1)
      endif

      call syncthreads()

      if (xnorm == 0.0_8) then
        if (tid == 1) then
          tau = 0.0_8
        endif
      else
        if (tid == 1) then
          xnorm = sqrt(xnorm)
          rv1 = abs(alphar)

          ! not taking abs of xnorm
          scal = max(rv1, xnorm)
          scal2 = min(rv1, xnorm)
          
          if (scal2 .eq. 0.0d0) then
            beta = -sign(scal, alphar)
          else
            beta = -sign(scal * sqrt(1.0d0 + (scal2/scal)**2), alphar)
          endif

          tau = (beta - alphar)/beta

          e = beta ! store beta in e vector
          alpha_s = 1.d0/(alphar - beta) !scaling factor for dscal
        endif

        call syncthreads()

        do i = tid, N, blockDim%x

          if (i <= N-1) then
            x(i) = alpha_s * x(i)
          elseif (i == N) then
            x(i) = 1.0_8
          endif
        
        end do

      endif

    end subroutine dlarfg_kernel

    attributes(global) subroutine dsyr2_mv_dlarfg_kernel(N, M, V, ldv, W, ldw, x, W2, ldw2, e, tau, finished)
      implicit none
      integer, value                                      :: N, M, ldv, ldw, ldw2
      real(8), dimension(1:ldv, 1:M), device, intent(in)  :: V
      real(8), dimension(1:ldw, 1:M), device, intent(in)  :: W
      real(8), dimension(1:ldw2, 2), device               :: W2
      real(8), dimension(1:N), device                     :: x
      real(8), device                                     :: tau
      real(8), device                                     :: e

      integer                                             :: i, j, tx, ty, tid, nb, laneid, istat, nBlocks
      integer, device                                     :: finished
      integer, shared                                     :: nFinished 
      real(8)                                             :: rv
      real(8)                                             :: rv1, rv2, rv3, scal, scal2, alphar, beta, rsum
      real(8), shared                                     :: xnorm
      real(8), shared                                     :: alpha_s

      tx = threadIdx%x
      ty = threadIdx%y
      i = (blockIdx%x - 1) * blockDim%x + tx
      j = (blockIdx%y - 1) * blockDim%y + ty

      nBlocks = gridDim%x * gridDim%y

      if (i <= N .and. j <= M) then

        rv = -W(N, j) * V(i,j) - V(N, j) * W(i,j)

        ! Update x
        istat = atomicadd(x(i), rv)
      endif

      if (ty == 1) then
        ! Zero out column for dgemv call
        if (i <= N) W2(i, 1) = 0
        ! Zero out workspace for intermediate dgemv results
        if (i <= M) then
          W2(N + i, 1) = 0
          W2(N + i, 2) = 0
        endif
      endif

      call threadfence()

      nFinished = 0
      call syncthreads()
      if (tx + ty == 2) nFinished = atomicinc(finished, nBlocks-1)
      call syncthreads()

      if ( nFinished < nBlocks - 1) return

      ! Begin dlarfg work with last block
      if (N == 1) return

      tid = tx + (ty - 1) * blockDim%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alpha_s = x(N-1)
        xnorm = 0.0_8
      endif

      call syncthreads()

      alphar = alpha_s
      rsum = 0.0_8

      nb = ceiling(real(N-1)/blockDim%x*blockDim%y) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= N-2) then
          rv1 = x(i)
          rv1 = rv1 * rv1
        else
          rv1 = 0.0_8
        endif

        rsum = rsum + rv1

        i = i + blockDim%x*blockDim%y
      end do

      ! Partial sum within warps using shuffle
      rv1 = rsum
      rv2 = __shfl_down(rv1,1)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,2)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,4)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,8)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,16)
      rv1 = rv1 + rv2

      if (laneID == 1) then
        istat = atomicadd(xnorm, rv1)
      endif

      call syncthreads()

      if (xnorm == 0.0_8) then
        if (tid == 1) then
          tau = 0.0_8
        endif
      else
        if (tid == 1) then
          xnorm = sqrt(xnorm)
          rv1 = abs(alphar)

          ! not taking abs of xnorm
          scal = max(rv1, xnorm)
          scal2 = min(rv1, xnorm)
          
          if (scal2 .eq. 0.0d0) then
            beta = -sign(scal, alphar)
          else
            beta = -sign(scal * sqrt(1.0d0 + (scal2/scal)**2), alphar)
          endif

          tau = (beta - alphar)/beta

          e = beta ! store beta in e vector
          alpha_s = 1.d0/(alphar - beta) !scaling factor for dscal
        endif

        call syncthreads()

        do i = tid, N-1, blockDim%x*blockDim%y

          if (i <= N-2) then
            x(i) = alpha_s * x(i)
          elseif (i == N-1) then
            x(i) = 1.0_8
          endif
        
        end do

      endif

    end subroutine dsyr2_mv_dlarfg_kernel

    attributes(global) subroutine stacked_dgemv_T(M, N, V, ldv, W, ldw, x, z1, z2)
      use cudafor
      implicit none
      integer, value                                  :: M, N, ldv, ldw
      real(8), dimension(ldv, M), device, intent(in)  :: V
      real(8), dimension(ldw, M), device, intent(in)  :: W
      real(8), dimension(N), device, intent(in)       :: x
      real(8), dimension(M), device                   :: z1, z2
      !complex(8), dimension(M), device, intent(in)        :: z1, z2

      !real(8), dimension(32), shared                     :: r_s
      !real(8), dimension(32), shared                     :: i_s

      integer :: i, j, tx, ty, istat
      real(8) :: rv1, rv2, xr

      tx = threadIdx%x
      ty = threadIdx%y

      i = (blockIdx%y - 1) * blockDim%y + ty 
      j = (blockIdx%x - 1) * blockDim%x + tx

      !if (i > 2*M .or. j > N) return
      if (i > 2*M) return

      xr = x(j)

      if (j > N) then
        rv1 = 0.d0
      else
        if (i > M) then
          rv2 = W(j, i-M)
        else
          rv2 = V(j, i)
        endif

        rv1 = rv2 * xr
      endif

      !Partial sum within warps using shuffle
      rv2 = __shfl_down(rv1,1)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,2)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,4)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,8)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,16)
      rv1 = rv1 + rv2

      if (tx == 1) then
        if (i > M) then
          istat = atomicadd(z2(i-M), rv1)
        else
          istat = atomicadd(z1(i), rv1)
        endif
      endif

      return
    end subroutine stacked_dgemv_T

    attributes(global) subroutine stacked_dgemv_N(M, N, V, ldv, W, ldw, z1, z2, y)
      use cudafor
      implicit none
      integer, value                                     :: M, N, ldv, ldw
      real(8), dimension(ldv, N), device, intent(in)     :: V
      real(8), dimension(ldw, N), device, intent(in)     :: W
      real(8), dimension(N), device, intent(in)          :: z1, z2
      real(8), dimension(M), device                      :: y

      integer :: i, j, tx, ty, istat
      real(8) :: rv1, rv2, xr

      tx = threadIdx%x
      ty = threadIdx%y

      i = (blockIdx%x - 1) * blockDim%x + tx
      j = (blockIdx%y - 1) * blockDim%y + ty

      if (i > M .or. j > 2*N) return

      if (j > N) then
        xr = z2(j-N)
        rv2 = V(i, j-N)
      else
        xr = z1(j)
        rv2 = W(i, j)
      endif

      rv1 = -rv2 * xr

      istat = atomicadd(y(i), rv1)

      return

    end subroutine stacked_dgemv_N

    attributes(global) subroutine finish_W_col_kernel(N, tau, x, y)
      implicit none
      integer, value                               :: N
      real(8), device                              :: tau
      real(8), dimension(N), device, intent(in)    :: x
      real(8), dimension(N), device                :: y

      integer                                      :: tid, i, j, k, nb, istat, laneID
      real(8)                                      :: rv1, rv2, rsum, mytau

      real(8), shared                              :: alphar
      !real(8), shared                              :: alpha
      real(8)                                      :: alpha

      tid = threadIdx%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alphar = 0.0_8
      endif

      call syncthreads()

      rsum = 0.0_8
      mytau = tau

      nb = ceiling(real(N)/blockDim%x) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= N) then
          rv1 = mytau * y(i) * x(i)
        else
          rv1 = 0.0d0
        endif

        rsum = rsum + rv1

        i = i + blockDim%x

      end do

      ! Partial sum within warps using shuffle
      rv1 = rsum
      rv2 = __shfl_down(rv1,1)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,2)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,4)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,8)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,16)
      rv1 = rv1 + rv2

      if (laneID == 1) then
        istat = atomicadd(alphar, rv1)
      endif

      call syncthreads()

      alpha = -0.5d0* mytau * alphar

      do i = tid, N, blockDim%x
        y(i) = mytau*y(i) + alpha * x(i) !daxpy
      end do

    end subroutine finish_W_col_kernel

    attributes(global) subroutine stacked_dgemv_N_finish_W(M, N, V, ldv, W, ldw, z1, z2, y, tau, x, finished)
      use cudafor
      implicit none
      integer, value                                     :: M, N, ldv, ldw
      real(8), dimension(ldv, N), device, intent(in)     :: V
      real(8), dimension(ldw, N), device, intent(in)     :: W
      real(8), dimension(N), device, intent(in)          :: z1, z2
      real(8), dimension(M), device                      :: y
      real(8), device                                    :: tau
      real(8), dimension(M), device, intent(in)          :: x
      integer, device                                    :: finished

      integer :: i, j, tx, ty, istat, nBlocks, tid, laneID, nb
      integer, shared :: nFinished
      real(8) :: rv1, rv2, rsum, xr, mytau
      real(8), shared                              :: alphar
      !real(8), shared                              :: alpha
      real(8)                                      :: alpha

      tx = threadIdx%x
      ty = threadIdx%y

      i = (blockIdx%x - 1) * blockDim%x + tx
      j = (blockIdx%y - 1) * blockDim%y + ty

      nBlocks = gridDim%x * gridDim%y

      if (i <= M .and. j <= 2*N) then

        if (j > N) then
          xr = z2(j-N)
          rv2 = V(i, j-N)
        else
          xr = z1(j)
          rv2 = W(i, j)
        endif

        rv1 = -rv2 * xr

        istat = atomicadd(y(i), rv1)
      endif

      call threadfence()

      nFinished = 0
      call syncthreads()
      if (tx + ty == 2) nFinished = atomicinc(finished, nBlocks-1)
      call syncthreads()

      if ( nFinished < nBlocks - 1) return

      ! Begin finish_W_col work with last block
      tid = threadIdx%x + (threadIdx%y - 1) * blockDim%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alphar = 0.0_8
      endif

      call syncthreads()

      rsum = 0.0_8
      mytau = tau

      nb = ceiling(real(M)/(blockDim%x*blockDim%y)) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= M) then
          rv1 = mytau * y(i) * x(i)
        else
          rv1 = 0.0d0
        endif

        rsum = rsum + rv1

        i = i + blockDim%x*blockDim%y

      end do

      ! Partial sum within warps using shuffle
      rv1 = rsum
      rv2 = __shfl_down(rv1,1)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,2)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,4)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,8)
      rv1 = rv1 + rv2
      rv2 = __shfl_down(rv1,16)
      rv1 = rv1 + rv2

      if (laneID == 1) then
        istat = atomicadd(alphar, rv1)
      endif

      call syncthreads()

      alpha = -0.5d0* mytau * alphar

      do i = tid, M, blockDim%x * blockDim%y
        y(i) = mytau*y(i) + alpha * x(i) !daxpy
      end do

    end subroutine stacked_dgemv_N_finish_W

end module dsytrd_gpu

