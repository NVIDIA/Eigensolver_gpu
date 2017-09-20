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

      kk = N-((N-1) / nb) * nb
      k = N+1
      do i = N-nb+1, kk+1+32, -nb
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

      do i = N, N-nb+1, -1
        iw = i-N+nb

        if (i < N) then
          blocks = ceiling(real(i)/threads)
          call dsyr2_mv_kernel<<<blocks, threads>>>(i, N-i, A(1, i+1), lda, W(1, iw+1), ldw, A(1, i), W(1, iw))
        end if

        if (i > 1) then
          ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)
          call dlarfg_kernel<<<1, threads>>>(i-1, e(i-1), A(1, i), tau(i-1))

          if (i < N) then
            istat = cublasSetStream(cuHandle, stream2)
            istat = cublasdgemv_v2(cuHandle, CUBLAS_OP_T, i-1, n-i, one, W(1, iw+1), ldw, A(1, i), 1, zero, W(i+1, iw+1), 1)
            istat = cublasdgemv_v2(cuHandle, CUBLAS_OP_N, i-1, n-i, -one, A(1, i+1), lda, W(i+1, iw+1), 1, zero, W(1, nb+1), 1)
            istat = cublasSetStream(cuHandle, stream3)
            istat =  cublasdgemv_v2(cuHandle, CUBLAS_OP_T, i-1, n-i, one, A(1, i+1), lda, A(1, i), 1, zero, W(i+1, iw), 1)
            istat = cublasdgemv_v2(cuHandle, CUBLAS_OP_N, i-1, n-i, -one, W(1, iw+1), ldw, W(i+1, iw), 1, zero, W(1, nb+2), 1)
            istat = cublasSetStream(cuHandle, stream1)
          else
            !$cuf kernel do(1) <<<*,*,0,stream2>>>
            do k = 1, i-1
              W(k,nb+1) = 0.d0
              W(k,nb+2) = 0.d0
            end do
          endif


          !Need to zero out vector in W if dsyr2_mv_kernel was not called previously
          if (i == N) then
            !$cuf kernel do(1) <<<*,*,0,stream1>>>
            do k = 1, i-1
              W(k,iw) = 0.d0
            end do
          endif

          blocks2D = dim3(10, ceiling(real(i-1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
          call dsymv_gpu<<<blocks2D, threads2D, 0, stream1>>>(i-1, A, lda, A(1, i), W(1, iw))

          call finish_W_col_kernel<<<1, threads>>>(i-1, tau(i-1), A(1, i), W(1, iw), W(1, nb+1), W(1, nb+2))

        end if
      end do
    end subroutine dlatrd_gpu

    attributes(global) subroutine dsyr2_mv_kernel(N, M, V, ldv, W, ldw, x, y)
      implicit none
      integer, value                                     :: N, M, ldv, ldw
      real(8), dimension(1:ldv, 1:M), device, intent(in) :: V
      real(8), dimension(1:ldw, 1:M), device, intent(in) :: W
      real(8), dimension(1:N), device                    :: x, y

      integer                                            ::  i, j
      real(8)                                            :: val

      i = (blockIdx%x - 1) * blockDim%x + threadIdx%x

      if (i > N) return

      ! Put x in registers, preserving real diagonal
      val = x(i)

      ! Perform x -= V * W(end,:)^T - W * V(end,:)^T
      do j = 1, M
        val = val - W(N, j) * V(i,j) - V(N, j) * W(i,j)
      end do

      ! Write x to global mem, preserving real diagonal
      x(i) = val

      ! Zero out column for zhemv call
      y(i) = 0

    end subroutine dsyr2_mv_kernel

    attributes(global) subroutine dlarfg_kernel(N, e, x, tau)
      implicit none
      integer, value                   :: N
      real(8), device                  :: tau
      real(8), device                  :: e
      real(8), dimension(N), device    :: x

      integer                          :: tid, i, j, nb, istat, laneID
      real(8)                          :: rv1, rv2, rv3, scal, scal2, alphar, beta, rsum
      real(8)                          :: cv1
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


        ! Partial sum within warps using shuffle
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

        rsum = rsum + rv1

        i = i + blockDim%x
      end do

      if (laneID == 1) then
        istat = atomicadd(xnorm, rsum)
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

        i = tid
        do j = 1, nb

          if (i <= N-1) then
            x(i) = alpha_s * x(i)
          elseif (i == N) then
            x(i) = 1.0_8
          endif
        
          i = i + blockDim%x
        end do

      endif

    end subroutine dlarfg_kernel

    attributes(global) subroutine finish_W_col_kernel(N, tau, x, y, y1, y2)
      implicit none
      integer, value                               :: N
      real(8), device                              :: tau
      real(8), dimension(N), device, intent(in)    :: x
      real(8), dimension(N), device                :: y, y1, y2

      integer                                      :: tid, i, j, k, nb, istat, laneID
      real(8)                                      :: rv1, rv2, iv1, iv2, rsum
      real(8)                                      :: val, cv1

      real(8), shared                              :: alphar
      real(8), shared                              :: alpha

      tid = threadIdx%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alphar = 0.0_8
      endif

      call syncthreads()

      rsum = 0.0_8

      nb = ceiling(real(N)/blockDim%x) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= N) then
          cv1 = tau*(y(i) + y1(i) + y2(i)) ! Add/scale intermediate results from previous calls
          val = cv1 * x(i)
          y(i) = cv1
        else
          val = 0.0d0
        endif

        rv1 = val

        ! Partial sum within warps using shuffle
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

        rsum = rsum + rv1

        i = i + blockDim%x

      end do

      if (laneID == 1) then
        istat = atomicadd(alphar, rsum)
      endif

      call syncthreads()

      if (tid == 1) then
        alpha = -0.5d0* tau * alphar
      end if

      call syncthreads()

      i = tid
      do j = 1, nb
        if (i <= N) then
          y(i) = y(i) + alpha * x(i) !daxpy
        endif
      
        i = i + blockDim%x
      end do

    end subroutine finish_W_col_kernel

    attributes(global) subroutine dsytd2_gpu(n,a,lda,d,e,tau)
      use cudafor
      implicit none
      integer, value    :: lda
      real(8),device    :: a(lda,32),tau(32)
      real(8),device    :: d(32),e(32)
      real(8),shared    :: a_s(32,32)
      real(8),shared    :: alpha
      real(8),shared    :: taui
      real(8)           :: beta
      real(8)           :: alphar
      real(8)           :: xnorm,x,y,z,w
      real(8)           :: wc
      integer, value    :: n
      integer           :: tx,ty,tl,i,j,ii

      tx=threadIdx%x
      ty=threadIdx%y
      ! Linear id of the thread (tx,ty)
      tl=tx+ blockDim%x*(ty-1)

      ! Load a_d in shared memory
      if (tx <= N .and. ty <= N) then
         a_s(tx           ,ty           )=a(tx           ,ty)
      endif

       call syncthreads()
      ! Symmetric matrix from upper triangular
      if (tx >ty) then
         a_s(tx,ty)=a_s(ty,tx)
      end if


      call syncthreads()

      ! For each column working backward
      do i=n-1,1,-1
        ! Generate elementary reflector
        ! Sum the vectors above the diagonal, only one warp active
        ! Reduce in a warp
        if (tl <=32) then
          if (tl <i) then
            w=a_s(tl,i+1)*a_s(tl,i+1)
          else 
            w=0._8
          endif

           xnorm=__shfl_down(w,1)
           w=w+xnorm
           xnorm=__shfl_down(w,2)
           w=w+xnorm
           xnorm=__shfl_down(w,4)
           w=w+xnorm
           xnorm=__shfl_down(w,8)
           w=w+xnorm
           xnorm=__shfl_down(w,16)
           w=w+xnorm
        end if

        if(tl==1) then
          alpha=a_s(i,i+1)
          alphar=dble(alpha)
          xnorm=dsqrt(w)
          
          if (xnorm .eq. 0_8) then
          ! H=1
            taui= 0._8
            alpha = 1.d0 ! To prevent scaling by dscal in this case
          else
            !Compute sqrt(alphar^2+xnorm^2) with  dlapy2(alphar,xnorm)
            x=abs(alphar)
            y=abs(xnorm)
            w=max(x,y)
            z=min(x,y)

            if (z .eq. 0.d0) then
              beta=-sign(w, alphar)
            else
              beta=-sign(w*sqrt(1.d0 + (z/w)**2), alphar)
            endif

            taui= (beta-alphar)/beta
            alpha = 1.d0/(alphar - beta) ! scale factor for dscal

          end if
        end if

        call syncthreads()

        ! dscal
        if (tl<i) then
          a_s(tl,i+1)=a_s(tl,i+1)*alpha
        end if

        if (tl==1) then 
          if (xnorm .ne. 0_8) then
            alpha=beta
          else
            alpha=a_s(i,i+1) ! reset alpha to original value
          endif

          e(i)=alpha
        end if

        if(taui.ne.(0.d0,0.d0)) then
          a_s(i,i+1)=1.d0
          call syncthreads()
          if(tl<=i) then
            tau(tl)=0.d0
            do j=1,i
              tau(tl)=tau(tl)+taui*a_s(tl,j)*a_s(j,i+1)
            end do
          end if

          call syncthreads()
       
          if (tl <=32) then
            if (tl <=i) then
              x=-.5d0*taui*tau(tl)*a_s(tl,i+1)
            else
              x=0._8
            endif

            z=__shfl_xor(x,1)
            x=x+z
            z=__shfl_xor(x,2)
            x=x+z
            z=__shfl_xor(x,4)
            x=x+z
            z=__shfl_xor(x,8)
            x=x+z
            z=__shfl_xor(x,16)
            x=x+z

          end if

          call syncthreads()

          if (tl <=i) then
            tau(tl)=tau(tl)+x*a_s(tl,i+1)
          end if

          if( tl==1) alpha=x

          call syncthreads()

          if( tx<=i .and. ty<=i) then
            a_s(tx,ty)=a_s(tx,ty)-a_s(tx,i+1)*tau(ty)-a_s(ty,i+1)*tau(tx)
          end if
          call syncthreads()

        endif

        if (tl==1) then
          a_s(i,i+1)=e(i)
          d(i+1)=a_s(i+1,i+1)
          tau(i)=taui
        end if

        call syncthreads()

      end do

      if (tl==1) then
        d(1) = a_s(1,1)
      endif


      call syncthreads()

      ! Back to device memory
      if (tx <= N .and. ty <= N) then
        a(tx,ty)=a_s(tx,ty)
      endif


    end subroutine dsytd2_gpu


end module dsytrd_gpu

