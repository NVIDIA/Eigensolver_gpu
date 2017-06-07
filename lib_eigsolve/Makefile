# Flags for GPUs with Pascal architecture. 
# If using Kepler, replace cc60 with cc35 and sm_60 with sm_35
FLAGS = -O3 -mp -pgf90libs -Mcuda=cc60,cuda8.0,ptxinfo,maxregcount:64 -Mlarge_arrays
CUFLAGS = -O3 -arch=sm_60
FLAGS2 = -O3 -mp -pgf90libs -Mcuda=cc60,cuda8.0,ptxinfo,maxregcount:64,nollvm -Mlarge_arrays

# Use this FLAGS2 if compiling on Power system
#FLAGS2 = -O1 -mp -pgf90libs -Mcuda=cc60,cuda8.0,ptxinfo,maxregcount:64,nollvm -Mlarge_arrays

# Uncomment to enable NVTX markers
#OPTFLAGS = -DUSE_NVTX

all: lib_eigsolve.a

OBJS = cusolverDn_m.o eigsolve_vars.o toolbox.o zhegst_gpu.o zhemv_gpu.o zhetrd_gpu.o zheevd_gpu.o zhegvdx_gpu.o 

zhetrd_gpu.o : zhetrd_gpu.F90
	pgf90 -c ${FLAGS2} ${OPTFLAGS}  $*.F90 -o $*.o
%.o: %.cuf
	pgf90 -c ${FLAGS} ${OPTFLAGS} $*.cuf -o $*.o
%.o: %.F90
	pgf90 -c ${FLAGS} ${OPTFLAGS} $*.F90 -o $*.o

lib_eigsolve.a: $(OBJS)
	ar -cr lib_eigsolve.a $(OBJS)

clean:
	rm lib_eigsolve.a *.mod *.o
