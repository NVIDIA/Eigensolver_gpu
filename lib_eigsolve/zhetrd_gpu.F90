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
  use zhemv_gpu

  contains
  
    subroutine zhetrd_gpu(uplo, N, A, lda, d, e, tau, work, lwork, nb)
      use eigsolve_vars
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

      kk = N-((N-1) / nb) * nb
      k = N+1
      do i = N-nb+1, kk+1+32, -nb
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
          !W(k,nb+1) = dcmplx(0,0)
          !W(k,nb+2) = dcmplx(0,0)
          W(k,iw) = dcmplx(0,0)
        end do

        blocks2D = dim3(10, ceiling(real(N-1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
        call zhemv_gpu<<<blocks2D, threads2D>>>(N-1, A, lda, A(1, N), W(1, iw))

        call finish_W_col_kernel<<<1, threads>>>(N-1, tau(N-1), A(1, N), W(1, iw), W(1, nb+1), W(1, nb+2))
      endif

      do i = N-1, N-nb+1, -1
        iw = i-N+nb

        blocks = ceiling(real(i)/threads)
        call zher2_mv_kernel<<<blocks, threads>>>(i, N-i, A(1, i+1), lda, W(1, iw+1), ldw, A(1, i), W(1, iw))

        if (i > 1) then
          ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)
          call zlarfg_kernel<<<1, threads>>>(i-1, e(i-1), A(1, i), tau(i-1))

          blocks2D = dim3(10, ceiling(real(i-1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
          call zhemv_gpu<<<blocks2D, threads2D>>>(i-1, A, lda, A(1, i), W(1, iw))

          ! TODO: Can eventually zero this out in a different kernel
          !$cuf kernel do(1) <<<*, *>>>
          do k = 1, n-i
            W(i+k, iw) = dcmplx(0,0)
            W(i+k, iw+1) = dcmplx(0,0)
          end do

          blocks2D = dim3(ceiling(real(i-1)/32), ceiling(real(2*(n-i))/8), 1)
          call stacked_zgemv_C<<<blocks2D, threads2D>>>(n-i, i-1, A(1,i+1), lda, W(1, iw+1), ldw, A(1,i), W(i+1, iw), W(i+1, iw+1))
          call stacked_zgemv_N<<<blocks2D, threads2D>>>(i-1, n-i, A(1,i+1), lda, W(1, iw+1), ldw, W(i+1,iw), W(i+1, iw+1), W(1, iw))

          call finish_W_col_kernel<<<1, threads>>>(i-1, tau(i-1), A(1, i), W(1, iw), W(1, nb+1), W(1, nb+2))

        end if
      end do
    end subroutine zlatrd_gpu

    attributes(global) subroutine zher2_mv_kernel(N, M, V, ldv, W, ldw, x, y)
      implicit none
      integer, value                                        :: N, M, ldv, ldw
      complex(8), dimension(1:ldv, 1:M), device, intent(in) :: V
      complex(8), dimension(1:ldw, 1:M), device, intent(in) :: W
      complex(8), dimension(1:N), device                    :: x, y

      integer                                               ::  i, j
      complex(8)                                            :: val

      i = (blockIdx%x - 1) * blockDim%x + threadIdx%x

      if (i > N) return

      ! Put x in registers, preserving real diagonal
      val = x(i)
      if (i == N) val = dble(val)

      ! Perform x -= V * W(end,:)^H - W * V(end,:)^H
      do j = 1, M
        val = val - conjg(W(N, j)) * V(i,j) - conjg(V(N, j)) * W(i,j)
      end do

      ! Write x to global mem, preserving real diagonal
      if (i == N) val = dble(val)
      x(i) = val

      ! Zero out column for zhemv call
      y(i) = 0

    end subroutine zher2_mv_kernel

    attributes(global) subroutine zlarfg_kernel(N, e, x, tau)
      implicit none
      integer, value                   :: N
      complex(8), device               :: tau
      real(8), device                  :: e
      complex(8), dimension(N), device :: x

      integer                          :: tid, i, j, nb, istat, laneID
      real(8)                          :: rv1, rv2, rv3, scal, alphar, alphai, beta, rsum, isum
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
          
          beta = -sign(scal * sqrt((rv1/scal)**2 + (rv2/scal)**2 + (xnorm/scal)**2), alphar)

          tau = dcmplx((beta - alphar)/beta, -alphai/beta)

          !zladiv
          rv1 = dble(alpha_s - beta)
          rv2 = dimag(alpha_s - beta)

          if (abs(rv2) .lt. abs(rv1)) then
            xnorm = rv2/rv1
            scal = rv1 + rv2*xnorm
            alpha_s = dcmplx(1.0_8/scal, -xnorm/scal)
          else
            xnorm = rv1/rv2
            scal = rv2 + rv1*xnorm
            alpha_s = dcmplx(xnorm/scal, -1.0_8/scal)
          endif

          e = beta ! store beta in e vector
        endif

        call syncthreads()

        i = tid
        do j = 1, nb

          if (i <= N-1) then
            x(i) = alpha_s * x(i)
          elseif (i == N) then
            !x(i) = 1.0_8
            x(i) = dcmplx(1.0_8, 0.0_8)
          endif
        
          i = i + blockDim%x
        end do

      endif

    end subroutine zlarfg_kernel

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

    attributes(global) subroutine finish_W_col_kernel(N, tau, x, y, y1, y2)
      implicit none
      integer, value                               :: N
      complex(8), device                           :: tau
      complex(8), dimension(N), device, intent(in) :: x
      complex(8), dimension(N), device             :: y, y1, y2

      integer                                      :: tid, i, j, k, nb, istat, laneID
      real(8)                                      :: rv1, rv2, iv1, iv2, rsum, isum
      complex(8)                                   :: val, cv1

      real(8), shared                              :: alphar, alphai
      complex(8), shared                           :: alpha

      tid = threadIdx%x
      laneID = iand(tid, 31)

      if (tid == 1) then
        alphar = 0.0_8
        alphai = 0.0_8
      endif

      call syncthreads()

      rsum = 0.0_8
      isum = 0.0_8

      nb = ceiling(real(N)/blockDim%x) ! number of blocks down column

      i = tid
      do j = 1, nb

        ! All threads perform their product, zero if out of bounds
        if (i <= N) then
          !cv1 = tau*(y(i) + y1(i) + y2(i)) ! Add/scale intermediate results from previous calls
          cv1 = tau * y(i) ! Add/scale intermediate results from previous calls
          val = dconjg(cv1) * x(i)
          y(i) = cv1
        else
          val = dcmplx(0.,0.)
        endif

        rv1 = dble(val); iv1 = dimag(val)

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

        isum = isum + iv1

        i = i + blockDim%x

      end do

      if (laneID == 1) then
        istat = atomicadd(alphar, rsum)
        istat = atomicadd(alphai, isum)
      endif

      call syncthreads()

      if (tid == 1) then
        alpha = -dcmplx(0.5, 0.0) * tau * dcmplx(alphar, alphai)

      end if

      call syncthreads()

      i = tid
      do j = 1, nb
        if (i <= N) then
          y(i) = y(i) + alpha * x(i) !zaxpy
        endif
      
        i = i + blockDim%x
      end do

    end subroutine finish_W_col_kernel

    attributes(global) subroutine zhetd2_gpu(n,a,lda,d,e,tau)
      use cudafor
      implicit none
      integer, value    :: lda
      complex(8),device :: a(lda,32),tau(32)
      real(8),device    :: d(32),e(32)
      complex(8),shared :: a_s(32,32)
      complex(8),shared :: alpha
      complex(8),shared :: taui
      real(8)           :: beta
      real(8)           :: alphar,alphai
      real(8)           :: xnorm,x,y,z,w
      complex(8)        :: wc
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
      ! Hermitian matrix from upper triangular
      if (tx >ty) then
         a_s(tx,ty)=conjg(a_s(ty,tx))
      end if

      ! Enforce diagonal element to be real
      if (tl==1) a_s(n,n)=dble(a_s(n,n))

      call syncthreads()

      ! For each column working backward
      do i=n-1,1,-1
        ! Generate elementary reflector
        ! Sum the vectors above the diagonal, only one warp active
        ! Reduce in a warp
        if (tl <=32) then
          if (tl <i) then
            w=a_s(tl,i+1)*conjg(a_s(tl,i+1))
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
          alphai=dimag(alpha)
          xnorm=dsqrt(w)
          
          if (xnorm .eq. 0_8 .and. alphai .eq. 0._8) then
          ! H=1
            taui= 0._8
            alpha = dcmplx(1.d0, 0.d0) ! To prevent scaling by zscal in this case
          else
            !Compute sqrt(alphar^2+alphai^2+xnorm^2) with  dlapy3(alphar,alphai,xnorm)
            x=abs(alphar)
            y=abs(alphai)
            z=abs(xnorm)
            w=max(x,y,z)
            beta=-sign(w*sqrt((x/w)**2+(y/w)**2+(z/w)**2),alphar)

            taui=dcmplx( (beta-alphar)/beta, -alphai/beta)

            !zladiv(dcmplx(one),alpha-beta)
            x= dble(alpha-beta)
            y=dimag(alpha-beta)
            if( abs(y) .lt. abs(x) ) then
              w=y/x
              z=x+y*w
              alpha=dcmplx(1/z,-w/z)
            else
              w=x/y
              z=y+x*w
              alpha=dcmplx(w/z,-1/z)
            end if
          end if
        end if

        call syncthreads()

        ! zscal
        if (tl<i) then
          a_s(tl,i+1)=a_s(tl,i+1)*alpha
        end if

        if (tl==1) then 
          if (xnorm .ne. 0_8 .or. alphai .ne. 0._8) then
            alpha=dcmplx(beta,0._8)
          else
            alpha=a_s(i,i+1) ! reset alpha to original value
          endif

          e(i)=alpha
        end if

        if(taui.ne.(0.d0,0.d0)) then
          a_s(i,i+1)=dcmplx(1.d0,0.d0)
          call syncthreads()
          if(tl<=i) then
          tau(tl)=dcmplx(0.d0,0.d0)
          do j=1,i
            tau(tl)=tau(tl)+taui*a_s(tl,j)*a_s(j,i+1)
          end do
        end if

        call syncthreads()
       
        if (tl <=32) then
          if (tl <=i) then
            wc=taui*conjg(tau(tl))*a_s(tl,i+1)
            x=-.5d0*dble(wc)
            y=-.5d0*dimag(wc)
          else
            x=0._8
            y=0._8
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

          w=__shfl_xor(y,1)
          y=y+w
          w=__shfl_xor(y,2)
          y=y+w
          w=__shfl_xor(y,4)
          y=y+w
          w=__shfl_xor(y,8)
          y=y+w
          w=__shfl_xor(y,16)
          y=y+w
        end if

       call syncthreads()

       if (tl <=i) then
         tau(tl)=tau(tl)+dcmplx(x,y)*a_s(tl,i+1)
       end if

       if( tl==1) alpha=dcmplx(x,y)

        call syncthreads()

        if( tx<=i .and. ty<=i) then
          a_s(tx,ty)=a_s(tx,ty)-a_s(tx,i+1)*dconjg(tau(ty))-dconjg(a_s(ty,i+1))*tau(tx)
        end if
        call syncthreads()

        else
          if(tl==1)a_s(i,i)=dble(a_s(i,i))
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


    end subroutine zhetd2_gpu


end module zhetrd_gpu

