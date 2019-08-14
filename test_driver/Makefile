CUDAROOT = ${CUDA_HOME}

# Flags for GPUs with Volta architecture.  Modify cc value as required.
FLAGS = -O3 -mp -pgf90libs -Mcuda=cc70,cuda10.1 -Mlarge_arrays

# extract PGI major version
PGI_MAJOR_VERSION := $(shell pgfortran -V | egrep -o '[0-9]+\.[0-9]+' | sed -e 's/\.[0-9]\+//g')

# PGI 19.1+ uses LLVM backend so MKL has to be linked differently
ifeq ($(shell expr ${PGI_MAJOR_VERSION} \< 19),1)
    MKL_THREAD=mkl_pgi_thread
else
    MKL_THREAD=mkl_intel_thread
endif

INCS = -I../lib_eigsolve 
LIBS = -Mcudalib=cublas,cusolver -L${CUDAROOT}/lib64 -lnvToolsExt
LIBS += ../lib_eigsolve/lib_eigsolve.a

# Populate this section for your LAPACK installation. Default here is for MKL.
LAPACKROOT = ${MKLROOT}
INCS += -I${LAPACKROOT}/include
LIBS += -L${LAPACKROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -l${MKL_THREAD} -pgf90libs -lpthread -lm -ldl

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
