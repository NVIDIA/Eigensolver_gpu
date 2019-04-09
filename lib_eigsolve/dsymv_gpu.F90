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

module dsymv_gpu
  use cudafor

  contains

#define BX 32
#define BY 8
#define NTILES 4

  attributes(global) subroutine dsymv_gpu(N, A, lda, x, y)
    use cudafor
    implicit none

    integer, value                                    :: N, lda
    real(8), dimension(lda, N), device, intent(in)    :: A
    real(8), dimension(N), device, intent(in)         :: x
    real(8), dimension(N), device                     :: y 

    real(8), dimension(BX+1, BX), shared              :: Ar_s
    real(8), dimension(BX), shared                    :: r_s

    integer                                           :: tx, ty, ii, jj, i, j, k, istat
    real(8)                                           :: rv1, rv2, mysum
    real(8)                                           :: Ar, xl

    ! ii,jj is index of top left corner of block
    ii = (blockIdx%y-1) * blockDim%x + 1

    mysum = 0.0_8

    tx = threadIdx%x
    ty = threadIdx%y

    if (ii + (blockIdx%x-1)*blockDim%x > N) return

    i = ii + tx - 1
    if (i <= N) then
      xl =  x(i) ! read part of x for lower triangular multiply
    endif

    ! Loop over columns (skip all lower triangular blocks)
    do jj = ii + (blockIdx%x-1)*blockDim%x, N, gridDim%x*blockDim%x
      j = jj + ty - 1

      ! Load block into shared memory
      ! CASE 1: Diagonal block
      if (ii == jj) then

        ! Load full block into shared memory
        do k = 0,NTILES-1
          if (i <= N .and. j + k * blockDim%y <= N) then
            Ar_s(tx, ty + k * blockDim%y) = A(i,j + k * blockDim%y)
          endif
        end do
        
        call syncthreads()

        ! Reflect to populate lower triangular part with true values of A
        do k = 0,NTILES-1
          if (tx > ty + k * blockDim%y) then
            Ar_s(tx, ty + k * blockDim%y) = Ar_s(ty + k * blockDim%y, tx)
          endif
        end do

        call syncthreads()

        do k = 0,NTILES-1
          if (i <= N .and. j + k * blockDim%y <= N ) then
            mysum = mysum + Ar_s(tx, ty + k * blockDim%y) * x(j + k*blockDim%y)
          endif
        end do

        !call syncthreads()

      ! CASE 2: Upper triangular block
      else if (ii < jj) then
        do k = 0,NTILES-1
          if (j + k * blockDim%y <= N) then
            Ar = A(i, j + k * blockDim%y)
          endif

          if (i <= N .and. j + k * blockDim%y <= N ) then
            mysum = mysum + Ar * x(j + k*blockDim%y)
          endif

          ! Perform product for symmetric lower block here
          if (i <= N .and. j + k*blockDim%y <= N) then
            rv1 = Ar * xl
          else
            rv1 = 0.0_8
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
            r_s(ty + k*blockDim%y) = rv1
          endif
        enddo

        call syncthreads()

        if (ty == 1 .and. jj+tx-1 <= N) then
          istat = atomicadd(y(jj + tx -1), r_s(tx))
        endif
        !call syncthreads()

      endif

      call syncthreads()

    end do

    if (i <= N) then
      istat = atomicadd(y(i), mysum)
    endif
    
  end subroutine dsymv_gpu

end module dsymv_gpu

