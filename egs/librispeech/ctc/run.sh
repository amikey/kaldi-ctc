#!/bin/bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
data=/data/LibriSpeech

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

stage=-2
nj=20
decode_nj=10
num_gpus=3

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

# you might not want to do this for interactive shells.
set -e


if [ $stage -le -2 ];then
  # download the data.  Note: we're using the 100 hour setup for
  # now; later in the script we'll download more and use it to train neural
  # nets.
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    local/download_and_untar.sh $data $data_url $part
  done

  # download the LM resources
  local/download_lm.sh $lm_url data/local/lm

  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done

  # when "--stage 3" option is used below we skip the G2P steps, and use the
  # lexicon we have already downloaded from openslr.org/11/
  local/prepare_dict.sh --stage 3 --nj $nj --cmd "$train_cmd" \
     data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh --share-silence-phones true --position-dependent-phones false \
    data/local/dict_nosp "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm

  # # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  # utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
  #   data/lang_nosp data/lang_nosp_test_tglarge
  # utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
  #   data/lang_nosp data/lang_nosp_test_fglarge
fi


if [ $stage -le -1 ];then
  utils/combine_data.sh \
    data/train_960 data/train_clean_100 data/train_clean_360 data/train_other_500 || exit 1;

  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc.conf \
    data/train_960 || exit 1;
  steps/compute_cmvn_stats.sh data/train_960 || exit 1;

  utils/subset_data_dir.sh data/train_960 100000 data/train_100k

  for test in test_clean test_other dev_clean dev_other; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc.conf \
      data/$test || exit 1;
    steps/compute_cmvn_stats.sh data/$test || exit 1;
  done
fi


if [ $stage -le 0 ];then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/train_100k data/lang_nosp exp/mono

  # decode using the monophone model
  (
    utils/mkgraph.sh --mono data/lang_nosp_test_tgsmall \
      exp/mono exp/mono/graph_nosp_tgsmall
    for test in test_clean test_other dev_clean dev_other; do
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/mono/graph_nosp_tgsmall \
        data/$test exp/mono/decode_nosp_tgsmall_$test
    done
  ) &
fi

if [ $stage -le 1 ];then
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/train_100k data/lang_nosp exp/mono exp/mono_ali_100k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      2000 10000 data/train_100k data/lang_nosp exp/mono_ali_100k exp/tri1

  # decode using the tri1 model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/tri1 exp/tri1/graph_nosp_tgsmall
    for test in test_clean test_other dev_clean dev_other; do
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri1/graph_nosp_tgsmall \
        data/$test exp/tri1/decode_nosp_tgsmall_$test
      # steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      #   data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
      # steps/lmrescore_const_arpa.sh \
      #   --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      #   data/$test exp/tri1/decode_nosp_{tgsmall,tglarge}_$test
    done
  ) &
fi


if [ $stage -le 2 ];then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train_100k data/lang_nosp exp/tri1 exp/tri1_ali_100k

  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
     --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
     data/train_100k data/lang_nosp exp/tri1_ali_100k exp/tri2b

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/tri2b exp/tri2b/graph_nosp_tgsmall
    for test in test_clean test_other dev_clean dev_other; do
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgsmall \
        data/$test exp/tri2b/decode_nosp_tgsmall_$test
      # steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
      #   data/$test exp/tri2b/decode_nosp_{tgsmall,tgmed}_$test
      # steps/lmrescore_const_arpa.sh \
      #   --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      #   data/$test exp/tri2b/decode_nosp_{tgsmall,tglarge}_$test
    done
  ) &
fi

wait


# The true spoken phoneme labels are obtained by aligning the audio and
# the alternative pronunciations with GMM acoustic model
bash run_ctc_phone.sh --stage 0 --num-gpus $num_gpus --minibatch-size 48 \
    --max-allow-frames 700 --frame-subsampling-factor 3 \
    --suffix "_fs3"

echo "$0: DONE."


