
all:
include ../kaldi.mk

EXTRA_CXXFLAGS = -Wno-sign-compare $(WARPCTC_FLAGS)

LDFLAGS += $(CUDA_LDFLAGS) $(WARPCTC_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS) $(WARPCTC_LDLIBS)

BINFILES = nnet-ctc-get-egs nnet-ctc-subset-egs nnet-ctc-copy-egs \
	nnet-ctc-shuffle-egs nnet-ctc-sort-egs nnet-ctc-relabel-egs nnet-ctc-perturb-egs \
	nnet2-ctc-train-simple \
	nnet2-ctc-compute-prob nnet2-ctc-compute-from-egs \
	nnet2-ctc-init-model ctc-make-decoding-graph nnet2-ctc-latgen-faster \
	ctc-graph-info

OBJFILES =

# Add this dependency to force cuda-compiled.o to be rebuilt when we reconfigure.
cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../ctc/kaldi-ctc.a ../nnet/kaldi-nnet.a ../nnet2/kaldi-nnet2.a ../gmm/kaldi-gmm.a \
         ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a  \
         ../transform/kaldi-transform.a ../tree/kaldi-tree.a \
         ../thread/kaldi-thread.a ../cudamatrix/kaldi-cudamatrix.a \
         ../matrix/kaldi-matrix.a ../fstext/kaldi-fstext.a \
         ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
