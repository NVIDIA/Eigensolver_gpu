module zhetd2_gpu
  contains
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
end module zhetd2_gpu
