## Makefile

.PHONY: clean realclean

FLAG_DNN_CUDA=1
FLAG_DNN_OMP=0

CUDA_PATH=/usr/local/cuda

### these flags can be modified
#OPTFLAG =-g
#OPTFLAG =-O3 -funroll-loops -ftree-vectorize -msse4 -ftree-vectorizer-verbose=1
#OPTFLAG =-O3 -funroll-loops
#OPTFLAG=-m64 -Ofast -flto -march=native -funroll-loops
OPTFLAG=-m64 -march=native -funroll-loops
CFLAGS= -lm

LIBNAME=libdnn
VER_MAJ=1
VER_MIN=0

BUILDDIR_A=../lib
BUILDDIR_SO=../lib
BUILDDIR_BIN=../bin
OBJDIR=../obj
SRCDIR=./src/
SRCDIR_HDRS=./inc
SRCDIR_CUDA=$(SRCDIR)

LIBS=-lm -lpthread
INCDIR=
LIBDIR=


### if everything goes fine, should not touch anything below

ifeq ($(FLAG_DNN_OMP),1) 
  DEFS+=-D__DNN_OMP
  CFLAGS+=-fopenmp
  LDFLAGS+=-fopenmp
endif

ifeq ($(FLAG_DNN_CUDA),1) 
  INCDIR+=$(CUDA_PATH)/include
  LIBDIR+=$(CUDA_PATH)/lib64
  LIBS+=-lcublas -lcuda -lcudart -lcurand
  DEFS+=-D__DNN_CUDA
  LDFLAGS+=-Wl,-rpath $(CUDA_PATH)/lib64
  NVCC=/usr/local/cuda/bin/nvcc
  OBJDIR_CUDA=$(OBJDIR)/cuda
  OBJS=$(patsubst $(SRCDIR_CUDA)/%.cu,$(OBJDIR_CUDA)/%.o, $(wildcard $(SRCDIR_CUDA)/*.cu))
  NVCCFLAGS=-Xcompiler -fPIC -arch=sm_30
endif

INCDIR+=$(SRCDIR_HDRS)
SONAME=$(LIBNAME).so.$(VER_MAJ)
OUTNAME=$(LIBNAME).so.$(VER_MAJ).$(VER_MIN)
OUTNAME_A=$(LIBNAME).a
OUTNAME_BIN=conv_gpu.out

INCFLAG+=$(patsubst %,-I%,$(INCDIR))
LDFLAGS+=$(patsubst %,-L%,$(LIBDIR))

CFLAGS+=-Wall -fPIC $(OPTFLAG)
LDFLAGS_SO=$(LDFLAGS) -shared -Wl,-soname,$(SONAME) $(OPTFLAG)
LDFLAGS_BIN=$(LDFLAGS) $(OPTFLAG)

HDRS=   $(wildcard $(SRCDIR_HDRS)/*.h)
OBJS+=   $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o, $(wildcard $(SRCDIR)/*.c))

CC=gcc
AR=ar

TARGET_SO=$(BUILDDIR_SO)/$(OUTNAME)
TARGET_A=$(BUILDDIR_A)/$(OUTNAME_A)
#TARGET_BIN=$(BUILDDIR_BIN)/$(OUTNAME_BIN)
TARGET_BIN=$(OUTNAME_BIN)


#all:$(TARGET_A) $(TARGET_SO)
all:$(TARGET_BIN)


$(TARGET_SO):$(OBJS) 
	@mkdir -p $(BUILDDIR_SO)
	$(CC) -o $(TARGET_SO)    $(LDFLAGS_SO) $(OBJS) $(LIBS)
	@ln -sf $(OUTNAME) $(BUILDDIR_SO)/$(SONAME)
	@ln -sf $(SONAME) $(BUILDDIR_SO)/$(LIBNAME).so

$(TARGET_A):$(OBJS)
	@mkdir -p $(BUILDDIR_A)
	@rm -f $(TARGET_A)
	$(AR) cr $(TARGET_A)    $(OBJS)

$(TARGET_BIN):$(OBJS)
	@mkdir -p $(BUILDDIR_BIN)
	$(CC) -o $(TARGET_BIN)    $(LDFLAGS_BIN) $(OBJS) $(LIBS)

# dependencies
$(OBJDIR)/%.o:$(SRCDIR)/%.c $(HDRS)
	@mkdir -p $(OBJDIR)
	$(CC) -o $@    $(DEFS) $(CFLAGS) $(INCFLAG) -c $<

$(OBJDIR_CUDA)/%.o:$(SRCDIR_CUDA)/%.cu $(HDRS)
	@mkdir -p $(OBJDIR_CUDA)
	$(NVCC) -o $@    $(DEFS) $(NVCCFLAGS) $(INCFLAG) -c $<

## other options
clean:
	clear
	rm -rf $(OBJS)

realclean:
	rm -rf $(OBJDIR) $(TARGET_SO) $(TARGET_A) $(TARGET_BIN) $(BUILDDIR_SO)/$(SONAME) $(BUILDDIR_SO)/$(LIBNAME).so

## end
