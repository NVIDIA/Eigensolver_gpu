CUDAROOT = /usr/local/cuda

# Flags for GPUs with Pascal architecture. 
# If using Kepler, replace cc60 with cc35 and sm_60 with sm_35
FLAGS = -O3 -mp -pgf90libs -Mcuda=cc60,cuda8.0,ptxinfo,maxregcount:64, -Mlarge_arrays

INCS = -I../lib_eigsolve 
LIBS = -Mcudalib=cublas,cusolver -L${CUDAROOT}/lib64 -lnvToolsExt
LIBS += ../lib_eigsolve/lib_eigsolve.a

# Populate this section for your LAPACK installation. Default here is for MKL.
LAPACKROOT = /opt/mkl
INCS += -I${LAPACKROOT}/include
LIBS += -L${LAPACKROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_pgi_thread -pgf90libs -lpthread -lm -ldl

# To enable comparison with MAGMA, uncomment and/or populate the following lines:
#MAGMAROOT = 
#INCS += -I${MAGMAROOT}/include
#LIBS += -L${MAGMAROOT}/lib -lmagma
#OPTFLAGS += -DHAVE_MAGMA

all: test_zhegvdx test_dsygvdx

test_zhegvdx: test_zhegvdx.F90 
	pgcc -c wallclock.c
	pgf90 -c ${FLAGS} toolbox.F90
	pgf90 -o test_zhegvdx test_zhegvdx.F90 toolbox.o wallclock.o ${LIBS} ${FLAGS} ${OPTFLAGS} -pgf90libs ${INCS}

test_dsygvdx: test_dsygvdx.F90 
	pgcc -c wallclock.c
	pgf90 -c ${FLAGS} toolbox.F90
	pgf90 -o test_dsygvdx test_dsygvdx.F90 toolbox.o wallclock.o ${LIBS} ${FLAGS} ${OPTFLAGS} -pgf90libs ${INCS}

clean:
	rm test_zhegvdx test_dsygvdx *.mod *.o
