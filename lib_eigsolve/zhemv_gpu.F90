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

module zhemv_gpu
  use cudafor

  contains

#define BX 32
#define BY 8
#define NTILES 4

  attributes(global) subroutine zhemv_gpu(N, A, lda, x, y)
    use cudafor
    implicit none

    integer, value                                    :: N, lda
    complex(8), dimension(lda, N), device, intent(in) :: A
    complex(8), dimension(N), device, intent(in)      :: x
    !DIR$ IGNORE_TKR y
    real(8), dimension(2*N), device                   :: y 

    real(8), dimension(BX+1, BX), shared              :: Ar_s
    real(8), dimension(BX+1, BX), shared              :: Ai_s
    real(8), dimension(BX), shared                    :: r_s
    real(8), dimension(BX), shared                    :: i_s

    integer                                           :: tx, ty, ii, jj, i, j, k, istat
    real(8)                                           :: rv1, rv2, iv1, iv2, myrsum, myisum
    real(8)                                           :: Ar, Ai, xrl, xil
    complex(8)                                        :: val

    ! ii,jj is index of top left corner of block
    ii = (blockIdx%y-1) * blockDim%x + 1
    !print*, "ii ", ii

    myrsum = 0.0_8
    myisum = 0.0_8

    tx = threadIdx%x
    ty = threadIdx%y

    if (ii + (blockIdx%x-1)*blockDim%x > N) return


    i = ii + tx - 1
    if (i <= N) then
      val =  x(i) ! read part of x for lower triangular multiply
    endif
    xrl = dble(val)
    xil = dimag(val)

    ! Loop over columns (skip all lower triangular blocks)
    do jj = ii + (blockIdx%x-1)*blockDim%x, N, gridDim%x*blockDim%x
      j = jj + ty - 1

      ! Load block into shared memory
      ! CASE 1: Diagonal block
      if (ii == jj) then

        ! Load full block into shared memory
        do k = 0,NTILES-1
          if (i <= N .and. j + k * blockDim%y <= N) then
            val = A(i, j + k*blockDim%y)
            Ar_s(tx, ty + k * blockDim%y) = dble(val)
            Ai_s(tx, ty + k * blockDim%y) = dimag(val)
          endif
        end do
        
        call syncthreads()

        ! Reflect to populate lower triangular part with true values of A
        do k = 0,NTILES-1
          if (tx > ty + k * blockDim%y) then
            Ar_s(tx, ty + k * blockDim%y) = Ar_s(ty + k * blockDim%y, tx)
            Ai_s(tx, ty + k * blockDim%y) = -Ai_s(ty + k * blockDim%y, tx)
          endif
        end do

        call syncthreads()

        do k = 0,NTILES-1
          if (i <= N .and. j + k * blockDim%y <= N ) then
            Ar = Ar_s(tx, ty + k * blockDim%y); Ai = Ai_s(tx, ty + k * blockDim%y)
            val = x(j + k*blockDim%y)
            rv1 = dble(val) ; iv1 = dimag(val)
            myrsum = myrsum + Ar * rv1 - Ai * iv1
            myisum = myisum + Ar * iv1 + Ai * rv1
          endif
        end do

        !call syncthreads()

      ! CASE 2: Upper triangular block
      else if (ii < jj) then
        do k = 0,NTILES-1
          if (j + k * blockDim%y <= N) then
            val = A(i, j + k * blockDim%y)
            Ar = dble(val)
            Ai = dimag(val)
          endif

          if (i <= N .and. j + k * blockDim%y <= N ) then
            val = x(j + k*blockDim%y)
            rv1 = dble(val) ; iv1 = dimag(val)
            myrsum = myrsum + Ar * rv1 - Ai * iv1
            myisum = myisum + Ar * iv1 + Ai * rv1
          endif

          ! Perform product for symmetric lower block here
          ! Don't need sync threads since thread is accessing own value
          !call syncthreads()
          if (i <= N .and. j + k*blockDim%y <= N) then
            rv1 = Ar * xrl + Ai * xil
            iv1 = Ar * xil - Ai * xrl
          else
            rv1 = 0.0_8
            iv1 = 0.0_8
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

          if (tx == 1) then
            i_s(ty + k*blockDim%y) = iv1
          endif
        enddo

        call syncthreads()

        if (ty == 1 .and. jj+tx-1 <= N) then
          istat = atomicadd(y(2*(jj + tx -1)-1), r_s(tx))
          istat = atomicadd(y(2*(jj + tx -1)), i_s(tx))
        endif
        !call syncthreads()

      endif

      call syncthreads()

    end do

    if (i <= N) then
      istat = atomicadd(y(2*i - 1), myrsum)
      istat = atomicadd(y(2*i), myisum)
    endif
    
  end subroutine zhemv_gpu

end module zhemv_gpu

