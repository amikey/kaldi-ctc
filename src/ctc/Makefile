
all:

include ../kaldi.mk

EXTRA_CXXFLAGS = -Wno-sign-compare $(WARPCTC_FLAGS)

LDFLAGS += $(CUDA_LDFLAGS) $(WARPCTC_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS) $(WARPCTC_LDLIBS)

TESTFILES =

OBJFILES = ctc-nnet-example.o ctc-nnet-update.o ctc-nnet-train.o ctc-graph.o \
	ctc-decoder-wrappers.o ctc-decodable-am-nnet.o

LIBNAME = kaldi-ctc

ADDLIBS =  ../nnet2/kaldi-nnet2.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../matrix/kaldi-matrix.a \
          ../cudamatrix/kaldi-cudamatrix.a ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

