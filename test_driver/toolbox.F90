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

module compare_utils

  interface compare
    module procedure compare_real_1d_cpu
    ! Note: compare_real/complex_2d_cpu compares absolute values in order to deal with
    !       eigenvectors which may differ only by sign
    module procedure compare_real_2d_cpu
    module procedure compare_complex_2d_cpu
  end interface compare

  contains

  subroutine compare_real_1d_cpu(A1_h,A2_h,N)
    implicit none
    real(8), dimension(:) :: A1_h, A2_h
    real(8), dimension(:), allocatable :: A1, A2
    real(8) :: maxerr,perr,l2normerr,norm,buf
    integer :: i,j,k,N,imax
    character (len=4) :: itcount
    character (len=1) :: proc

    allocate(A1, source = A1_h)
    allocate(A2, source = A2_h)

    l2normerr = 0.d0
    norm = 0.d0
    maxerr = 0.d0
    imax=1
    do i=1, N
        if(abs(A1(i)) >= 1e-10) then
          perr = abs(A1(i) - A2(i))/abs(A1(i))*100.d0
          norm = norm + abs(A1(i)*A1(i));
          l2normerr = l2normerr + abs((A1(i) - A2(i))*(A1(i) - A2(i)))
        else
          perr = 0.d0
        endif
        if(perr>maxerr .and. A1(i)/=0.0d0 .and. A2(i)/=0.0d0) then
          maxerr = perr
          imax = i
        endif
    enddo

  norm = sqrt(norm)
  l2normerr = sqrt(l2normerr)
  if(l2normerr /= 0.d0) then
    l2normerr = l2normerr/norm
    write(*,"(A16,2X,ES10.3,A12,ES10.3,A6,I5,A6,2X,E20.14,2X,A6,2X,E20.14)") &
    "l2norm error",l2normerr,"max error",maxerr,"% at",imax,"cpu=",A1(imax),"gpu=",A2(imax)
  else
    write(*,"(A16)") "EXACT MATCH"
  endif

  deallocate(A1, A2)

  end subroutine compare_real_1d_cpu

  subroutine compare_real_2d_cpu(A1_h,A2_h,N, M)
    implicit none
    real(8), dimension(:,:) :: A1_h, A2_h
    real(8), dimension(:,:), allocatable :: A1, A2
    real(8) :: maxerr,perr,l2normerr,norm,buf
    integer :: i,j,k,imax,jmax,kmax,N,M
    character (len=4) :: itcount
    character (len=1) :: proc

    allocate(A1, source = A1_h)
    allocate(A2, source = A2_h)

    l2normerr = 0.d0
    norm = 0.d0
    maxerr = 0.d0
    imax=1
    jmax=1
    kmax=1
    do j=1, M
      do i=1, N
        if(abs(A1(i,j)) >= 1e-10) then
          perr = abs(abs(A1(i,j)) - abs(A2(i,j)))/abs(A1(i,j))*100.d0
          norm = norm + abs(A1(i,j)*A1(i,j));
          l2normerr = l2normerr + abs((abs(A1(i,j)) - abs(A2(i,j)))*(abs(A1(i,j)) - abs(A2(i,j))))
        else
          perr = 0.d0
        endif
        if(perr>maxerr .and. A1(i,j)/=0.0d0 .and. A2(i,j)/=0.0d0) then
          maxerr = perr
          imax = i
          jmax = j
        endif
      enddo
   enddo

  norm = sqrt(norm)
  l2normerr = sqrt(l2normerr)
  if(l2normerr /= 0.d0) then
    l2normerr = l2normerr/norm
    write(*,"(A16,2X,ES10.3,A12,ES10.3,A6,I5,I5,A6,2X,E20.14,1X,2X,A6,2X,E20.14,1X)") &
    "l2norm error",l2normerr,"max error",maxerr,"% at",imax,jmax,"cpu=",REAL(A1(imax,jmax)),"gpu=",REAL(A2(imax,jmax))
  else
    write(*,"(A16)") "EXACT MATCH"
  endif

  deallocate(A1, A2)

  end subroutine compare_real_2d_cpu

  subroutine compare_complex_2d_cpu(A1_h,A2_h,N, M)
    implicit none
    complex(8), dimension(:,:) :: A1_h, A2_h
    complex(8), dimension(:,:), allocatable :: A1, A2
    real(8) :: maxerr,perr,l2normerr,norm,buf
    integer :: i,j,k,imax,jmax,kmax,N,M
    character (len=4) :: itcount
    character (len=1) :: proc

    allocate(A1, source = A1_h)
    allocate(A2, source = A2_h)

    l2normerr = 0.d0
    norm = 0.d0
    maxerr = 0.d0
    imax=1
    jmax=1
    kmax=1
    do j=1, M
      do i=1, N
        if(abs(A1(i,j)) >= 1e-10) then
          perr = abs(abs(A1(i,j)) - abs(A2(i,j)))/abs(A1(i,j))*100.d0
          norm = norm + abs(A1(i,j)*A1(i,j));
          l2normerr = l2normerr + abs((abs(A1(i,j)) - abs(A2(i,j)))*(abs(A1(i,j)) - abs(A2(i,j))))
        else
          perr = 0.d0
        endif
        if(perr>maxerr .and. A1(i,j)/=0.0d0 .and. A2(i,j)/=0.0d0) then
          maxerr = perr
          imax = i
          jmax = j
        endif
      enddo
   enddo

  norm = sqrt(norm)
  l2normerr = sqrt(l2normerr)
  if(l2normerr /= 0.d0) then
    l2normerr = l2normerr/norm
    write(*,"(A16,2X,ES10.3,A12,ES10.3,A6,I5,I5,A6,2X,E20.14,1X,E20.14,2X,A6,2X,E20.14,1X,E20.14)") &
    "l2norm error",l2normerr,"max error",maxerr,"% at",imax,jmax,"cpu=",REAL(A1(imax,jmax)),AIMAG(A1(imax,jmax)),"gpu=",REAL(A2(imax,jmax)),AIMAG(A2(imax,jmax))
  else
    write(*,"(A16)") "EXACT MATCH"
  endif

  deallocate(A1, A2)

  end subroutine compare_complex_2d_cpu
end module compare_utils
