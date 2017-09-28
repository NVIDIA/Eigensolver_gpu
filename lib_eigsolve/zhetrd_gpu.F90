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

module zhetrd_gpu
  use cudafor
  use cublas

  contains
  
    subroutine zhetrd_gpu(uplo, N, A, lda, d, e, tau, work, lwork, nb)
      use eigsolve_vars
      use zhetd2_gpu
      implicit none
      character                                 :: uplo
      integer                                   :: N, lda, lwork, nb, nx, ldwork, istat
      integer                                   :: i, j, k, kk
      real(8), dimension(1:N), device           :: d
      real(8), dimension(1:N-1), device         :: e
      complex(8), dimension(1:lwork), device    :: work
      complex(8), dimension(1:lda, 1:N), device :: A
      complex(8), dimension(1:N-1), device      :: tau
      complex(8), parameter                     :: cone = cmplx(1,0,8)
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
        call zlatrd_gpu(uplo, i+nb-1, nb, A, lda, e, tau, work, ldwork)

        ! Update trailing submatrix
        call cublaszher2k(uplo, 'N', i-1, nb, -cone, A(1, i), lda, work, ldwork, one, a, lda)

        k = k - nb

      end do
      
      ! Finish any remaining columns to get final 32x32 block
      nb = k - 32 - 1
      i = k - nb
      
      if (nb > 0) then
        ! Reduce columns i:i+nb-1 to tridiagonal form 
        call zlatrd_gpu(uplo, i+nb-1, nb, A, lda, e, tau, work, ldwork)

        ! Update trailing submatrix
        call cublaszher2k(uplo, 'N', i-1, nb, -cone, A(1, i), lda, work, ldwork, one, a, lda)
      endif

      ! Final block
      threads = dim3(32, 32, 1)
      call zhetd2_gpu<<<1, threads>>>(min(32, N), A, lda, d, e, tau)

      ! Copy superdiagonal back into A, store diagonal in d
      !$cuf kernel do(1) <<<*,*>>>
      do j = 33, N
        !A(j-1, j) = e(j-1) ! JR Not strictly needed so skipping this copy
        d(j) = A(j,j)
      end do

    end subroutine zhetrd_gpu


    subroutine zlatrd_gpu(uplo, N, nb, A, lda, e, tau, W, ldw)
      use eigsolve_vars
      use zhemv_gpu
      implicit none
      character                                  :: uplo
      integer                                    :: N, nb, lda, ldw, istat
      integer                                    :: i, j, k, iw
      integer                                    :: blocks, threads
      complex(8), dimension(1:lda, 1:N), device  :: A
      complex(8), dimension(1:ldw, 1:nb), device :: W
      complex(8), dimension(N-1), device         :: tau
      real(8), dimension(N-1), device            :: e
      complex(8), parameter                      :: cone = cmplx(1, 0, 8), czero = cmplx(0, 0, 8), chalf = cmplx(0.5, 0, 8)

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
        call zlarfg_kernel<<<1, threads>>>(N-1, e(N-1), A(1, N), tau(N-1))

        !$cuf kernel do(1) <<<*,*>>>
        do k = 1, N-1
          W(k,iw) = dcmplx(0,0)
        end do

        blocks2D = dim3(10, ceiling(real(N-1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
        call zhemv_gpu<<<blocks2D, threads2D>>>(N-1, A, lda, A(1, N), W(1, iw))

        call finish_W_col_kernel<<<1, threads>>>(N-1, tau(N-1), A(1, N), W(1, iw))
      endif

      do i = N-1, N-nb+1, -1
        iw = i-N+nb

        blocks2D = dim3(ceiling(real(max(i, N-i))/32), ceiling(real(N-i)/8), 1)
        !call zher2_mv_kernel<<<blocks2D, threads2D>>>(i, N-i, A(1, i+1), lda, W(1, iw+1), ldw, A(1, i), W(1, iw), ldw)
        call zher2_mv_zlarfg_kernel<<<blocks2D, threads2D>>>(i, N-i, A(1, i+1), lda, W(1, iw+1), ldw, A(1, i), W(1, iw), ldw, e(i-1), tau(i-1), A(1, i), finished(1))

        if (i > 1) then
          ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)
          !call zlarfg_kernel<<<1, threads>>>(i-1, e(i-1), A(1, i), tau(i-1))

          blocks2D = dim3(min(10, ceiling(real(i-1)/32)), ceiling(real(i-1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
          call zhemv_gpu<<<blocks2D, threads2D>>>(i-1, A, lda, A(1, i), W(1, iw))

          blocks2D = dim3(ceiling(real(i-1)/32), ceiling(real(2*(n-i))/8), 1)
          call stacked_zgemv_C<<<blocks2D, threads2D>>>(n-i, i-1, A(1,i+1), lda, W(1, iw+1), ldw, A(1,i), W(i+1, iw), W(i+1, iw+1))
          !call stacked_zgemv_N<<<blocks2D, threads2D>>>(i-1, n-i, A(1,i+1), lda, W(1, iw+1), ldw, W(i+1,iw), W(i+1, iw+1), W(1, iw))
          call stacked_zgemv_N_finish_W<<<blocks2D, threads2D>>>(i-1, n-i, A(1,i+1), lda, W(1, iw+1), ldw, W(i+1,iw), W(i+1, iw+1), W(1, iw), tau(i-1), A(1, i), W(1, iw), finished(1))

          !call finish_W_col_kernel<<<1, threads>>>(i-1, tau(i-1), A(1, i), W(1, iw))

        end if
      end do
    end subroutine zlatrd_gpu

    attributes(global) subroutine zher2_mv_kernel(N, M, V, ldv, W, ldw, x, W2, ldw2)
      implicit none
      integer, value                                        :: N, M, ldv, ldw, ldw2
      complex(8), dimension(1:ldv, 1:M), device, intent(in) :: V
      complex(8), dimension(1:ldw, 1:M), device, intent(in) :: W
      complex(8), dimension(1:ldw2, 2), device               :: W2
      !DIR$ IGNORE_TKR x
      real(8), dimension(1:2*N), device                     :: x

      integer                                               :: i, j, istat
      complex(8)                                            :: val
      real(8)                                               :: rv, iv

      i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
      j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

      if (i <= N .and. j <= M) then

        val = - conjg(W(N, j)) * V(i,j) - conjg(V(N, j)) * W(i,j)
        rv = dble(val)
        iv = dimag(val)

        ! Zero out imaginary part on diagonal
        if (i == N) then
          iv = 0.d0
        endif

        ! Update x
        istat = atomicadd(x(2*i -1), rv)
        istat = atomicadd(x(2*i), iv)
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

    end subroutine zher2_mv_kernel

    attributes(global) subroutine zlarfg_kernel(N, e, x, tau)
      implicit none
      integer, value                   :: N
      complex(8), device               :: tau
      real(8), device                  :: e
      complex(8), dimension(N), device :: x

      integer                          :: tid, i, j, nb, istat, laneID
      real(8)                          :: rv1, rv2, rv3, scal, invscal, alphar, alphai, beta, rsum, isum
      complex(8)                       :: cv1
      real(8), shared                  :: xnorm
      complex(8), shared               :: alpha_s

      tid = threadIdx%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alpha_s = x(N)
        xnorm = 0.0_8
      endif

      call syncthreads()

      alphar = dble(alpha_s)
      alphai = dimag(alpha_s)
      rsum = 0.0_8

      nb = ceiling(real(N)/blockDim%x) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= N-1) then
          cv1 = x(i)
          rv2 = dble(cv1); rv3 = dimag(cv1)
          rv1 = rv2*rv2 + rv3*rv3
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

      if (xnorm == 0.0_8 .and. alphai == 0.0_8) then
        if (tid == 1) then
          tau = 0.0_8
        endif
      else
        if (tid == 1) then
          xnorm = sqrt(xnorm)

          rv1 = abs(alphar)
          rv2 = abs(alphai)
          ! not taking abs of xnorm
          scal = max(rv1, rv2, xnorm)
          invscal = 1.d0/scal

          rv1 = rv1 * invscal
          rv2 = rv2 * invscal
          xnorm = xnorm * invscal
          
          beta = -sign(scal * sqrt(rv1*rv1 + rv2*rv2 + xnorm*xnorm), alphar)

          tau = dcmplx((beta - alphar)/beta, -alphai/beta)

          !zladiv
          rv1 = dble(alpha_s - beta)
          rv2 = dimag(alpha_s - beta)

          if (abs(rv2) .lt. abs(rv1)) then
            xnorm = rv2/rv1
            invscal = 1.d0/(rv1 + rv2*xnorm)
            alpha_s = dcmplx(invscal, -xnorm * invscal)
          else
            xnorm = rv1/rv2
            invscal = 1.d0/(rv2 + rv1*xnorm)
            alpha_s = dcmplx(xnorm * invscal, -invscal)
          endif

          e = beta ! store beta in e vector
        endif

        call syncthreads()

        do i = tid, N, blockDim%x
          cv1 = x(i)

          if (i <= N-1) then
            cv1 = alpha_s * cv1
          elseif (i == N) then
            !x(i) = 1.0_8
            cv1 = dcmplx(1.0_8, 0.0_8)
          endif

          x(i) = cv1
        end do

      endif

    end subroutine zlarfg_kernel

    attributes(global) subroutine zher2_mv_zlarfg_kernel(N, M, V, ldv, W, ldw, x, W2, ldw2, e, tau, x2, finished)
      implicit none
      integer, value                                        :: N, M, ldv, ldw, ldw2
      complex(8), dimension(1:ldv, 1:M), device, intent(in) :: V
      complex(8), dimension(1:ldw, 1:M), device, intent(in) :: W
      complex(8), dimension(1:ldw2, 2), device              :: W2
      !DIR$ IGNORE_TKR x
      real(8), dimension(1:2*N), device                     :: x
      complex(8), dimension(1:N), device                    :: x2
      complex(8), device                                    :: tau
      real(8), device                                       :: e

      integer                                               :: i, j, tx, ty, tid, nb, laneid, istat, nBlocks
      integer, device                                       :: finished
      integer, shared                                       :: nFinished 
      complex(8)                                            :: val
      real(8)                                               :: rv, iv
      real(8)                                               :: rv1, rv2, rv3, scal, invscal, alphar, alphai, beta, rsum, isum
      complex(8)                                            :: cv1
      real(8), shared                                       :: xnorm
      complex(8), shared                                    :: alpha_s

      tx = threadIdx%x
      ty = threadIdx%y
      i = (blockIdx%x - 1) * blockDim%x + tx
      j = (blockIdx%y - 1) * blockDim%y + ty

      nBlocks = gridDim%x * gridDim%y

      !if (i > N .or. j > M) return
      if (i <= N .and. j <= M) then

        val = - conjg(W(N, j)) * V(i,j) - conjg(V(N, j)) * W(i,j)
        rv = dble(val)
        iv = dimag(val)

        ! Zero out imaginary part on diagonal
        if (i == N) then
          iv = 0.d0
        endif

        ! Update x
        istat = atomicadd(x(2*i -1), rv)
        istat = atomicadd(x(2*i), iv)
      endif

      if (ty == 1) then
        ! Zero out column for zhemv call
        if (i <= N) W2(i, 1) = 0
        ! Zero out workspace for intermediate zgemv results
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

      ! Begin zlarfg work with last block
      if (N == 1) return

      tid = tx + (ty - 1) * blockDim%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alpha_s = x2(N-1)
        xnorm = 0.0_8
      endif

      call syncthreads()

      alphar = dble(alpha_s)
      alphai = dimag(alpha_s)
      rsum = 0.0_8

      nb = ceiling(real(N-1)/(blockDim%x*blockDim%y)) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= N-2) then
          cv1 = x2(i)
          rv2 = dble(cv1); rv3 = dimag(cv1)
          rv1 = rv2*rv2 + rv3*rv3
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

      if (xnorm == 0.0_8 .and. alphai == 0.0_8) then
        if (tid == 1) then
          tau = 0.0_8
        endif
      else
        if (tid == 1) then
          xnorm = sqrt(xnorm)

          rv1 = abs(alphar)
          rv2 = abs(alphai)
          ! not taking abs of xnorm
          scal = max(rv1, rv2, xnorm)
          invscal = 1.d0/scal

          rv1 = rv1 * invscal
          rv2 = rv2 * invscal
          xnorm = xnorm * invscal
          
          beta = -sign(scal * sqrt(rv1*rv1 + rv2*rv2 + xnorm*xnorm), alphar)

          tau = dcmplx((beta - alphar)/beta, -alphai/beta)

          !zladiv
          rv1 = dble(alpha_s - beta)
          rv2 = dimag(alpha_s - beta)

          if (abs(rv2) .lt. abs(rv1)) then
            xnorm = rv2/rv1
            invscal = 1.d0/(rv1 + rv2*xnorm)
            alpha_s = dcmplx(invscal, -xnorm * invscal)
          else
            xnorm = rv1/rv2
            invscal = 1.d0/(rv2 + rv1*xnorm)
            alpha_s = dcmplx(xnorm * invscal, -invscal)
          endif

          e = beta ! store beta in e vector
        endif

        call syncthreads()

        do i = tid, N-1, blockDim%x*blockDim%y
          cv1 = x2(i)

          if (i <= N-2) then
            cv1 = alpha_s * cv1
          elseif (i == N-1) then
            !x(i) = 1.0_8
            cv1 = dcmplx(1.0_8, 0.0_8)
          endif

          x2(i) = cv1
        end do

      endif

    end subroutine zher2_mv_zlarfg_kernel

    attributes(global) subroutine stacked_zgemv_C(M, N, V, ldv, W, ldw, x, z1, z2)
      use cudafor
      implicit none
      integer, value                                     :: M, N, ldv, ldw
      complex(8), dimension(ldv, M), device, intent(in)  :: V
      complex(8), dimension(ldw, M), device, intent(in)  :: W
      complex(8), dimension(N), device, intent(in)       :: x
      !DIR$ IGNORE_TKR z1, z2
      real(8), dimension(2*M), device                    :: z1, z2
      !complex(8), dimension(M), device, intent(in)        :: z1, z2

      !real(8), dimension(32), shared                     :: r_s
      !real(8), dimension(32), shared                     :: i_s

      integer :: i, j, tx, ty, istat
      complex(8) :: val
      real(8) :: rv1, rv2, iv1, iv2, xr, xi

      tx = threadIdx%x
      ty = threadIdx%y

      i = (blockIdx%y - 1) * blockDim%y + ty 
      j = (blockIdx%x - 1) * blockDim%x + tx

      !if (i > 2*M .or. j > N) return
      if (i > 2*M) return

      val = x(j)
      xr = dble(val); xi = dimag(val)

      if (j > N) then
        !val = dcmplx(0,0)
        rv1 = 0.d0; iv1 = 0.d0
      else
        if (i > M) then
          val = W(j, i-M)
        else
          val = V(j, i)
        endif

        rv2 = dble(val); iv2 = dimag(val)

        rv1 = rv2 * xr + iv2 * xi
        iv1 = rv2 * xi - iv2 * xr
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

      !if (tx == 1) then
        !r_s(ty + k*blockDim%y) = rv1
        !r_s(ty) = rv1
      !endif

      !Partial sum within warps using shuffle
      iv2 = __shfl_down(iv1,1)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,2)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,4)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,8)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,16)
      iv1 = iv1 + iv2

      !if (tx == 1) then
        !i_s(ty + k*blockDim%y) = iv1
        !i_s(ty) = iv1
      !endif

      !call syncthreads()

      !if (ty == 1 .and. i+tx-1 <= 2*M) then
      !  if (i+tx-1 > M) then
      !    istat = atomicadd(z2(2*(i+tx-1-M) - 1), r_s(tx))
      !    istat = atomicadd(z2(2*(i+tx-1-M)), i_s(tx))
      !  else
      !    istat = atomicadd(z1(2*(i+tx-1) - 1), r_s(tx))
      !    istat = atomicadd(z1(2*(i+tx-1)), i_s(tx))
      !  endif
      !endif

      if (tx == 1) then
        if (i > M) then
          istat = atomicadd(z2(2*(i-M) - 1), rv1)
          istat = atomicadd(z2(2*(i-M)), iv1)
        else
          istat = atomicadd(z1(2*i - 1), rv1)
          istat = atomicadd(z1(2*i), iv1)
        endif
      endif

      return
    end subroutine stacked_zgemv_C

    attributes(global) subroutine stacked_zgemv_N(M, N, V, ldv, W, ldw, z1, z2, y)
      use cudafor
      implicit none
      integer, value                                     :: M, N, ldv, ldw
      complex(8), dimension(ldv, N), device, intent(in)  :: V
      complex(8), dimension(ldw, N), device, intent(in)  :: W
      complex(8), dimension(N), device, intent(in)       :: z1, z2
      !DIR$ IGNORE_TKR y
      real(8), dimension(2*M), device                    :: y

      integer :: i, j, tx, ty, istat
      complex(8) :: val1, val2
      real(8) :: rv1, rv2, iv1, iv2, xr, xi

      tx = threadIdx%x
      ty = threadIdx%y

      i = (blockIdx%x - 1) * blockDim%x + tx
      j = (blockIdx%y - 1) * blockDim%y + ty

      if (i > M .or. j > 2*N) return

      if (j > N) then
        val1 = z2(j-N)
        val2 = V(i, j-N)
      else
        val1 = z1(j)
        val2 = W(i, j)
      endif
      xr = dble(val1); xi = dimag(val1)
      rv2 = dble(val2); iv2 = dimag(val2)

      rv1 = -rv2 * xr + iv2 * xi
      iv1 = -rv2 * xi - iv2 * xr

      istat = atomicadd(y(2*i-1), rv1)
      istat = atomicadd(y(2*i), iv1)

      return

    end subroutine stacked_zgemv_N

    attributes(global) subroutine finish_W_col_kernel(N, tau, x, y)
      implicit none
      integer, value                               :: N
      complex(8), device                           :: tau
      complex(8), dimension(N), device, intent(in) :: x
      complex(8), dimension(N), device             :: y

      integer                                      :: tid, i, j, k, nb, istat, laneID
      real(8)                                      :: rv1, rv2, iv1, iv2, rsum, isum
      complex(8)                                   :: val, cv1, mytau

      real(8), shared                              :: alphar, alphai
      !complex(8), shared                          :: alpha
      complex(8)                                   :: alpha

      tid = threadIdx%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alphar = 0.0_8
        alphai = 0.0_8
      endif

      call syncthreads()

      rsum = 0.0_8
      isum = 0.0_8
      mytau = tau

      nb = ceiling(real(N)/blockDim%x) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= N) then
          val = dconjg(mytau * y(i)) * x(i)
        else
          val = dcmplx(0.,0.)
        endif

        rv1 = dble(val); iv1 = dimag(val)

        rsum = rsum + rv1
        isum = isum + iv1

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

      iv1 = isum
      iv2 = __shfl_down(iv1,1)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,2)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,4)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,8)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,16)
      iv1 = iv1 + iv2

      if (laneID == 1) then
        istat = atomicadd(alphar, rv1)
        istat = atomicadd(alphai, iv1)
      endif

      call syncthreads()

      alpha = -dcmplx(0.5, 0.0) * mytau * dcmplx(alphar, alphai)

      do i = tid, N, blockDim%x
        y(i) = mytau*y(i) + alpha * x(i) !zaxpy
      end do

    end subroutine finish_W_col_kernel

    attributes(global) subroutine stacked_zgemv_N_finish_W(M, N, V, ldv, W, ldw, z1, z2, y, tau, x, y2, finished)
      use cudafor
      implicit none
      integer, value                                     :: M, N, ldv, ldw
      complex(8), dimension(ldv, N), device, intent(in)  :: V
      complex(8), dimension(ldw, N), device, intent(in)  :: W
      complex(8), dimension(N), device, intent(in)       :: z1, z2
      !DIR$ IGNORE_TKR y
      real(8), dimension(2*M), device                    :: y
      complex(8), device                                 :: tau
      complex(8), dimension(M), device, intent(in)       :: x
      complex(8), dimension(M), device                   :: y2
      integer, device                                    :: finished

      integer :: i, j, tx, ty, istat, nBlocks, tid, laneID, nb
      integer, shared :: nFinished
      complex(8) :: val1, val2, mytau, alpha
      real(8) :: rv1, rv2, iv1, iv2, xr, xi, rsum, isum
      real(8), shared :: alphar, alphai

      tx = threadIdx%x
      ty = threadIdx%y

      i = (blockIdx%x - 1) * blockDim%x + tx
      j = (blockIdx%y - 1) * blockDim%y + ty

      nBlocks = gridDim%x * gridDim%y

      if (i <= M .and. j <= 2*N) then
        if (j > N) then
          val1 = z2(j-N)
          val2 = V(i, j-N)
        else
          val1 = z1(j)
          val2 = W(i, j)
        endif
        xr = dble(val1); xi = dimag(val1)
        rv2 = dble(val2); iv2 = dimag(val2)

        rv1 = -rv2 * xr + iv2 * xi
        iv1 = -rv2 * xi - iv2 * xr

        istat = atomicadd(y(2*i-1), rv1)
        istat = atomicadd(y(2*i), iv1)
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
        alphai = 0.0_8
      endif

      call syncthreads()

      rsum = 0.0_8
      isum = 0.0_8
      mytau = tau

      nb = ceiling(real(M)/(blockDim%x * blockDim%y)) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= M) then
          val1 = dconjg(mytau * y2(i)) * x(i)
        else
          val1 = dcmplx(0.,0.)
        endif

        rv1 = dble(val1); iv1 = dimag(val1)

        rsum = rsum + rv1
        isum = isum + iv1

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

      iv1 = isum
      iv2 = __shfl_down(iv1,1)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,2)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,4)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,8)
      iv1 = iv1 + iv2
      iv2 = __shfl_down(iv1,16)
      iv1 = iv1 + iv2

      if (laneID == 1) then
        istat = atomicadd(alphar, rv1)
        istat = atomicadd(alphai, iv1)
      endif

      call syncthreads()

      alpha = -dcmplx(0.5, 0.0) * mytau * dcmplx(alphar, alphai)

      do i = tid, M, blockDim%x * blockDim%y
        y2(i) = mytau*y2(i) + alpha * x(i) !zaxpy
      end do

    end subroutine stacked_zgemv_N_finish_W


end module zhetrd_gpu

